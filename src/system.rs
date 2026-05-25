use reqwest::Client;
use std::process::Command;

use crate::types::{ModelInfo, OllamaShow, OllamaTags, OllamaVersion, SystemInfo};

/// Run a CLI command and return its stdout, or a fallback string on failure.
pub fn run_cmd(cmd: &str, args: &[&str], fallback: &str) -> String {
    match Command::new(cmd).args(args).output() {
        Ok(out) => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        Err(_) => fallback.to_string(),
    }
}

/// Detect the GPU by trying nvidia-smi first, then lspci for AMD/Intel.
pub fn detect_gpu() -> (String, bool) {
    // Try NVIDIA
    let nvidia = Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output();
    if let Ok(out) = nvidia {
        let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !name.is_empty() && !name.contains("[FAILED]") {
            return (format!("NVIDIA: {}", name), true);
        }
    }

    // Try GPU via lspci
    let lspci = Command::new("lspci").arg("-nn").output();
    if let Ok(out) = lspci {
        let output = String::from_utf8_lossy(&out.stdout);
        for line in output.lines() {
            if line.to_lowercase().contains("vga") || line.to_lowercase().contains("3d") {
                // Extract description after the ": " separator
                let desc = line.split(": ").nth(1).unwrap_or(&line);
                // Strip PCI metadata: [hex:hex], [hex], (rev X)
                let clean = desc
                    .split_whitespace()
                    .filter(|tok| {
                        !(tok.starts_with('[') && tok.ends_with(']')) && !tok.starts_with("(rev")
                    })
                    .collect::<Vec<&str>>()
                    .join(" ");
                let lower = clean.to_lowercase();
                // Strip vendor prefix for shorter display
                let short = if let Some(comma) = clean.find(',') {
                    clean[comma + 1..].trim().to_string()
                } else {
                    clean.clone()
                };
                if lower.contains("amd") || lower.contains("radeon") {
                    return (format!("AMD: {}", short), true);
                }
                if lower.contains("intel") {
                    return (format!("Intel: {}", short), true);
                }
                return (clean, true);
            }
        }
    }

    // Try macOS GPU
    let system_profiler = Command::new("system_profiler")
        .arg("SPDisplaysDataType")
        .output();
    if let Ok(out) = system_profiler {
        let output = String::from_utf8_lossy(&out.stdout);
        for line in output.lines() {
            if line.contains("Chipset Model") || line.contains("GPU Model") {
                return (
                    line.trim()
                        .trim_start_matches("Chipset Model:")
                        .trim_start_matches("GPU Model:")
                        .trim()
                        .to_string(),
                    true,
                );
            }
        }
    }

    ("CPU-only (no GPU detected)".to_string(), false)
}

/// Gather system + Ollama info before benchmarking.
pub async fn gather_system_info(client: &Client, host: &str, model_name: &str) -> SystemInfo {
    // OS
    let os = run_cmd("uname", &["-srm"], "Unknown OS");

    // CPU
    let cpu = if cfg!(target_os = "linux") {
        run_cmd(
            "grep",
            &["-m1", "model name", "/proc/cpuinfo"],
            "Unknown CPU",
        )
        .trim_start_matches("model name\t:")
        .trim()
        .to_string()
    } else if cfg!(target_os = "macos") {
        run_cmd("sysctl", &["-n", "machdep.cpu.brand_string"], "Unknown CPU")
    } else {
        run_cmd("wmic", &["cpu", "get", "name"], "Unknown CPU")
    };

    // RAM
    let ram_total = if cfg!(target_os = "linux") {
        let meminfo = run_cmd(
            "awk",
            &["/MemTotal/{print $2, $3}", "/proc/meminfo"],
            "Unknown",
        );
        if meminfo != "Unknown" {
            let kb: f64 = meminfo
                .split_whitespace()
                .next()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0);
            if kb > 1_000_000.0 {
                format!("{:.0} GiB", kb / 1_048_576.0)
            } else {
                format!("{:.0} MiB", kb / 1024.0)
            }
        } else {
            "Unknown".to_string()
        }
    } else if cfg!(target_os = "macos") {
        run_cmd("sysctl", &["-n", "hw.memsize"], "Unknown")
    } else {
        "Unknown".to_string()
    };

    // GPU
    let (gpu, _has_gpu) = detect_gpu();

    // Ollama version
    let ollama_version = match client.get(format!("{}/api/version", host)).send().await {
        Ok(resp) => resp
            .json::<OllamaVersion>()
            .await
            .unwrap_or_else(|_| OllamaVersion {
                version: "unknown".to_string(),
            }),
        Err(_) => OllamaVersion {
            version: "unreachable".to_string(),
        },
    };

    // Model details from /api/tags
    let tags = match client.get(format!("{}/api/tags", host)).send().await {
        Ok(resp) => match resp.json::<OllamaTags>().await {
            Ok(t) => Some(t),
            Err(_) => None,
        },
        Err(_) => None,
    };

    let model_info: Option<ModelInfo> = tags.and_then(|t| {
        t.models.into_iter().find(|m| {
            m.name == model_name
                || m.name
                    .strip_suffix(":latest")
                    .map(|n| n == model_name)
                    .unwrap_or(false)
        })
    });

    let (model_params, model_quant, model_family, model_size) = if let Some(mi) = model_info {
        (
            mi.details
                .parameter_size
                .clone()
                .unwrap_or_else(|| "N/A".to_string()),
            mi.details
                .quantization_level
                .clone()
                .unwrap_or_else(|| "N/A".to_string()),
            {
                let fam = mi
                    .details
                    .families
                    .clone()
                    .filter(|v| !v.is_empty())
                    .map(|v| v.join(", "));
                fam.or_else(|| mi.details.family.clone())
                    .unwrap_or_else(|| "N/A".to_string())
            },
            format!("{:.1} GB", mi.size as f64 / 1_073_741_824.0),
        )
    } else {
        (
            "N/A".to_string(),
            "N/A".to_string(),
            "N/A".to_string(),
            "N/A".to_string(),
        )
    };

    // Determine device: if GPU detected and not CPU-only, assume GPU inference
    let device = if _has_gpu {
        "GPU".to_string()
    } else {
        "CPU".to_string()
    };

    // KV Cache type from /api/show
    let kv_cache_type = match client
        .post(format!("{}/api/show", host))
        .json(&serde_json::json!({ "name": model_name }))
        .send()
        .await
    {
        Ok(resp) => match resp.json::<OllamaShow>().await {
            Ok(show) => show
                .projectors
                .and_then(|projs| projs.into_iter().find_map(|p| p.cache_type))
                .unwrap_or_else(|| "N/A".to_string()),
            Err(_) => "N/A".to_string(),
        },
        Err(_) => "N/A".to_string(),
    };

    SystemInfo {
        os,
        cpu,
        ram_total,
        gpu,
        ollama_version: ollama_version.version,
        model_params,
        model_quant,
        model_family,
        model_size,
        device,
        kv_cache_type,
    }
}
