use reqwest::Client;
use std::process::Command;

use crate::types::{OllamaShow, OllamaVersion, SystemInfo};
use crate::utils::{fetch_tags, ModelDetailsPlain};

/// Run a CLI command and return its stdout, or a fallback string on failure.
fn run_cmd(cmd: &str, args: &[&str], fallback: &str) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|_| fallback.to_string())
}

/// Detect the GPU by trying nvidia-smi first, then lspci, then system_profiler.
fn detect_gpu() -> (String, bool) {
    // Try NVIDIA
    if let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
    {
        let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !name.is_empty() && !name.contains("[FAILED]") {
            return (format!("NVIDIA: {}", name), true);
        }
    }

    // Try GPU via lspci
    if let Ok(out) = Command::new("lspci").arg("-nn").output() {
        let output = String::from_utf8_lossy(&out.stdout);
        for line in output.lines() {
            let lower = line.to_lowercase();
            if !(lower.contains("vga") || lower.contains("3d")) {
                continue;
            }
            let desc = line.split(": ").nth(1).unwrap_or(line);
            // Strip PCI metadata: [hex:hex], [hex], (rev X)
            let clean: String = desc
                .split_whitespace()
                .filter(|tok| {
                    !(tok.starts_with('[') && tok.ends_with(']')) && !tok.starts_with("(rev")
                })
                .collect::<Vec<&str>>()
                .join(" ");
            let lower_clean = clean.to_lowercase();
            // Strip vendor prefix for shorter display
            let short = clean
                .find(',')
                .map(|comma| clean[comma + 1..].trim().to_string())
                .unwrap_or_else(|| clean.clone());
            if lower_clean.contains("amd") || lower_clean.contains("radeon") {
                return (format!("AMD: {}", short), true);
            }
            if lower_clean.contains("intel") {
                return (format!("Intel: {}", short), true);
            }
            return (clean, true);
        }
    }

    // Try macOS GPU
    if let Ok(out) = Command::new("system_profiler")
        .arg("SPDisplaysDataType")
        .output()
    {
        let output = String::from_utf8_lossy(&out.stdout);
        for line in output.lines() {
            if line.contains("Chipset Model") || line.contains("GPU Model") {
                let name = line
                    .trim()
                    .trim_start_matches("Chipset Model:")
                    .trim_start_matches("GPU Model:")
                    .trim()
                    .to_string();
                return (name, true);
            }
        }
    }

    ("CPU-only (no GPU detected)".to_string(), false)
}

/// Gather system + Ollama info before benchmarking.
pub async fn gather_system_info(
    client: &Client,
    host: &str,
    model_name: &str,
    ctx: u32,
    iterations: usize,
) -> SystemInfo {
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
        let meminfo = run_cmd("awk", &["/MemTotal/{print $2, $3}", "/proc/meminfo"], "Unknown");
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
        let raw = run_cmd("sysctl", &["-n", "hw.memsize"], "Unknown");
        raw.parse::<f64>()
            .map(|b| format!("{:.0} GiB", b / 1_073_741_824.0))
            .unwrap_or(raw)
    } else {
        "Unknown".to_string()
    };

    // GPU
    let (gpu, has_gpu) = detect_gpu();

    // Ollama version
    let ollama_version = match client
        .get(format!("{}/api/version", host))
        .send()
        .await
    {
        Ok(resp) => match resp.json::<OllamaVersion>().await {
            Ok(v) => v.version,
            Err(_) => "unknown".to_string(),
        },
        Err(_) => "unreachable".to_string(),
    };

    // Model info
    let details = match fetch_tags(client, host).await {
        Some(tags) => tags
            .models
            .iter()
            .find(|m| {
                m.name == model_name
                    || m.name
                        .strip_suffix(":latest")
                        .map(|n| n == model_name)
                        .unwrap_or(false)
            })
            .map(ModelDetailsPlain::from_model_info)
            .unwrap_or_else(ModelDetailsPlain::na),
        None => ModelDetailsPlain::na(),
    };

    // Device: if GPU detected and not CPU-only, assume GPU inference
    let device = if has_gpu { "GPU" } else { "CPU" };

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
        ollama_version,
        model_name: model_name.to_string(),
        model_params: details.params,
        model_quant: details.quant,
        model_family: details.family,
        model_size: details.size,
        device: device.to_string(),
        kv_cache_type,
        ctx,
        iterations,
    }
}
