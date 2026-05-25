use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::process::Command;

const DEFAULT_MODEL: &str = "qwen3.5:9b-q4_K_M";
const DEFAULT_HOST: &str = "http://localhost:11434";

// Simulate a heavy repository file analysis prompt
const DEFAULT_PROMPT: &str = "Analyze the architectural structure of this code block, check for any concurrency bottlenecks, and propose an idiomatic refactor.";

#[derive(Parser, Debug)]
#[command(author, version, about = "Ollama Rust Hardware Benchmark Suite")]
struct Args {
    /// Model to benchmark (single mode)
    #[arg(short, long, default_value_t = std::borrow::Cow::Borrowed(DEFAULT_MODEL))]
    model: std::borrow::Cow<'static, str>,

    /// Models to compare (comparison mode; pass 2+ models)
    #[arg(short = 'C', long, num_args = 2..)]
    compare: Vec<String>,

    /// Number of benchmark iterations
    #[arg(short, long, default_value_t = 3)]
    iterations: usize,

    /// Max tokens to generate per trial (limits decode time)
    #[arg(long, default_value_t = 256)]
    num_predict: u32,

    /// Context window size in tokens
    #[arg(short, long, default_value_t = 24576)]
    ctx: u32,

    /// Temperature for generation
    #[arg(short, long, default_value_t = 0.2)]
    temperature: f32,

    /// Custom prompt (uses default if omitted)
    #[arg(long)]
    prompt: Option<String>,

    /// Ollama API host URL
    #[arg(long, default_value_t = std::borrow::Cow::Borrowed(DEFAULT_HOST))]
    host: std::borrow::Cow<'static, str>,
}

#[derive(Deserialize, Debug)]
struct OllamaResponse {
    prompt_eval_count: u32,
    prompt_eval_duration: u64, // nanoseconds
    eval_count: u32,
    eval_duration: u64, // nanoseconds
}

#[derive(Serialize)]
struct OllamaOptions {
    num_ctx: u32,
    num_predict: u32,
    temperature: f32,
}

#[derive(Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    options: OllamaOptions,
}

// --- System info structs ---

#[derive(Deserialize, Debug)]
struct OllamaVersion {
    version: String,
}

#[derive(Deserialize, Debug)]
struct ModelDetails {
    #[allow(dead_code)]
    format: Option<String>,
    family: Option<String>,
    families: Option<Vec<String>>,
    parameter_size: Option<String>,
    quantization_level: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ModelInfo {
    name: String,
    size: u64,
    details: ModelDetails,
}

#[derive(Deserialize, Debug)]
struct ProjectorInfo {
    name: String,
    size: u64,
    digest: String,
    cache_type: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OllamaShow {
    #[allow(dead_code)]
    parameters: Option<String>,
    projectors: Option<Vec<ProjectorInfo>>,
}

#[derive(Deserialize, Debug)]
struct OllamaTags {
    models: Vec<ModelInfo>,
}

struct SystemInfo {
    os: String,
    cpu: String,
    ram_total: String,
    gpu: String,
    ollama_version: String,
    model_params: String,
    model_quant: String,
    model_family: String,
    model_size: String,
    device: String, // "GPU" or "CPU"
    kv_cache_type: String,
}

struct RunMetrics {
    prefill_tps: f64,
    decode_tps: f64,
}

struct ModelBenchmarkResult {
    model: String,
    avg_prefill: f64,
    min_prefill: f64,
    max_prefill: f64,
    avg_decode: f64,
    min_decode: f64,
    max_decode: f64,
    params: String,
    quant: String,
    family: String,
    size: String,
}

/// Run a CLI command and return its stdout, or a fallback string on failure.
fn run_cmd(cmd: &str, args: &[&str], fallback: &str) -> String {
    match Command::new(cmd).args(args).output() {
        Ok(out) => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        Err(_) => fallback.to_string(),
    }
}

/// Detect the GPU by trying nvidia-smi first, then lspci for AMD/Intel.
fn detect_gpu() -> (String, bool) {
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
                    clean[comma + 1..].trim()
                } else {
                    &clean
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
async fn gather_system_info(client: &Client, host: &str, model_name: &str) -> SystemInfo {
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

async fn run_trial(
    client: &Client,
    model: &str,
    prompt: &str,
    ctx: u32,
    num_predict: u32,
    temperature: f32,
    host: &str,
) -> Result<RunMetrics, Box<dyn std::error::Error>> {
    let payload = OllamaRequest {
        model,
        prompt,
        stream: false,
        options: OllamaOptions {
            num_ctx: ctx,
            num_predict,
            temperature,
        },
    };

    let res: OllamaResponse = client
        .post(&format!("{}/api/generate", host))
        .json(&payload)
        .send()
        .await?
        .json()
        .await?;

    let prefill_sec = res.prompt_eval_duration as f64 / 1_000_000_000.0;
    let decode_sec = res.eval_duration as f64 / 1_000_000_000.0;

    Ok(RunMetrics {
        prefill_tps: res.prompt_eval_count as f64 / prefill_sec,
        decode_tps: res.eval_count as f64 / decode_sec,
    })
}

/// Benchmark a single model and return aggregated stats.
async fn benchmark_model(
    client: &Client,
    model: &str,
    prompt: &str,
    iterations: usize,
    ctx: u32,
    num_predict: u32,
    temperature: f32,
    host: &str,
) -> Option<ModelBenchmarkResult> {
    // Fetch model metadata
    let tags = match client.get(format!("{}/api/tags", host)).send().await {
        Ok(resp) => match resp.json::<OllamaTags>().await {
            Ok(t) => Some(t),
            Err(_) => None,
        },
        Err(_) => None,
    };

    let model_info: Option<ModelInfo> = tags.and_then(|t| {
        t.models.into_iter().find(|m| {
            m.name == model
                || m.name
                    .strip_suffix(":latest")
                    .map(|n| n == model)
                    .unwrap_or(false)
        })
    });

    let (params, quant, family, size) = if let Some(mi) = model_info {
        (
            mi.details
                .parameter_size
                .clone()
                .unwrap_or_else(|| "N/A".to_string()),
            mi.details
                .quantization_level
                .clone()
                .unwrap_or_else(|| "N/A".to_string()),
            mi.details
                .families
                .clone()
                .filter(|v| !v.is_empty())
                .map(|v| v.join(", "))
                .or_else(|| mi.details.family.clone())
                .unwrap_or_else(|| "N/A".to_string()),
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

    // Warmup
    let _ = run_trial(
        client,
        model,
        "Hello, world!",
        ctx,
        num_predict,
        temperature,
        host,
    )
    .await;

    let mut prefill_results = Vec::new();
    let mut decode_results = Vec::new();

    let pb = ProgressBar::new(iterations as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:30.cyan/blue}] {pos}/{len} ({msg})",
        )
        .unwrap()
        .progress_chars("#>-"),
    );
    pb.set_message(format!("Benchmarking {}", model));

    for _ in 0..iterations {
        match run_trial(client, model, prompt, ctx, num_predict, temperature, host).await {
            Ok(metrics) => {
                prefill_results.push(metrics.prefill_tps);
                decode_results.push(metrics.decode_tps);
            }
            Err(e) => eprintln!("\n  ⚠️  Trial failed for {}: {}", model, e),
        }
        pb.inc(1);
    }
    pb.finish_and_clear();

    if prefill_results.is_empty() {
        println!("  ❌ All trials failed for {}. Skipping.", model);
        return None;
    }

    Some(ModelBenchmarkResult {
        model: model.to_string(),
        avg_prefill: prefill_results.iter().sum::<f64>() / prefill_results.len() as f64,
        min_prefill: prefill_results.iter().cloned().fold(f64::NAN, f64::min),
        max_prefill: prefill_results.iter().cloned().fold(f64::NAN, f64::max),
        avg_decode: decode_results.iter().sum::<f64>() / decode_results.len() as f64,
        min_decode: decode_results.iter().cloned().fold(f64::NAN, f64::min),
        max_decode: decode_results.iter().cloned().fold(f64::NAN, f64::max),
        params,
        quant,
        family,
        size,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let client = Client::new();
    let prompt = args.prompt.as_deref().unwrap_or(DEFAULT_PROMPT);

    // Print system info header
    let sys = gather_system_info(&client, &args.host, &args.model).await;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  🚀  Ollama Rust Hardware Benchmark Suite                    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  System Info                                                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  OS:             {}", sys.os);
    println!("║  CPU:            {}", sys.cpu);
    println!("║  RAM:            {}", sys.ram_total);
    println!("║  GPU:            {}", sys.gpu);
    println!("║  Inference:      {}", sys.device);
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Benchmark Config                                            ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Context Window:  {} tokens", args.ctx);
    println!("║  Iterations:      {} runs (+ 1 warmup)", args.iterations);
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // --- Comparison mode ---
    if !args.compare.is_empty() {
        println!(
            "🔍  Comparison mode: benchmarking {} models\n",
            args.compare.len()
        );

        let mut results: Vec<ModelBenchmarkResult> = Vec::new();
        for model in &args.compare {
            print!("  ▶  Benchmarking {} ... ", model);
            match benchmark_model(
                &client,
                model,
                prompt,
                args.iterations,
                args.ctx,
                args.num_predict,
                args.temperature,
                &args.host,
            )
            .await
            {
                Some(r) => {
                    results.push(r);
                    println!("OK");
                }
                None => println!("FAILED"),
            }
        }

        if results.is_empty() {
            println!("All models failed. Verify Ollama is running and models are pulled.");
            return Ok(());
        }

        // Print comparison table
        println!(
            "\n╔══════════════════════════════════════════════════════════════════════════════════════╗"
        );
        println!(
            "║  📊  Model Comparison Results                                                             ║"
        );
        println!(
            "╠══════════════════════════════════════════════════════════════════════════════════════╣"
        );

        // Header
        println!(
            "║  {:<42}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8} ║",
            "Model", "Params", "AvgPre", "MinPre", "MaxPre", "AvgDec", "MaxDec"
        );
        println!(
            "╠{:─<42}╬{:─<9}╬{:─<9}╬{:─<9}╬{:─<9}╬{:─<9}╬{:─<9}╣",
            "", "", "", "", "", "", ""
        );

        for r in &results {
            println!(
                "║  {:<42}  {:>8}  {:>8.1}  {:>8.1}  {:>8.1}  {:>8.1}  {:>8.1} ║",
                r.model,
                r.params,
                r.avg_prefill,
                r.min_prefill,
                r.max_prefill,
                r.avg_decode,
                r.max_decode
            );
        }

        println!(
            "╚══════════════════════════════════════════════════════════════════════════════════════╝"
        );

        // Per-model detail cards
        for r in &results {
            println!("\n╔══════════════════════════════════════════════════════════════╗");
            println!(
                "║  📋  {}                                                      ║",
                r.model
            );
            println!("╠════════════════════════════════════════════════════════════════╣");
            println!("║  Architecture:   {}", r.family);
            println!("║  Parameters:     {}", r.params);
            println!("║  Quantization:   {}", r.quant);
            println!("║  Model Size:     {}", r.size);
            println!("╠════════════════════════════════════════════════════════════════╣");
            println!(
                "║  Prefill (tokens/s)   Avg: {:>10.2}  Min: {:>10.2}  Max: {:>10.2}",
                r.avg_prefill, r.min_prefill, r.max_prefill
            );
            println!(
                "║  Decode  (tokens/s)   Avg: {:>10.2}  Min: {:>10.2}  Max: {:>10.2}",
                r.avg_decode, r.min_decode, r.max_decode
            );
            println!("╚════════════════════════════════════════════════════════════════╝");
        }

        return Ok(());
    }

    // --- Single model mode ---
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Ollama & Model                                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Ollama Version:  {}", sys.ollama_version);
    println!("║  Model:          {}", args.model);
    println!("║  Parameters:     {}", sys.model_params);
    println!("║  Quantization:   {}", sys.model_quant);
    println!("║  Architecture:   {}", sys.model_family);
    println!("║  Model Size:     {}", sys.model_size);
    println!("║  KV Cache Type:  {}", sys.kv_cache_type);
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Warmup
    print!("⏳ Priming {} (Warmup run)... ", sys.device);
    let _ = run_trial(
        &client,
        &args.model,
        "Hello, world!",
        args.ctx,
        args.num_predict,
        args.temperature,
        &args.host,
    )
    .await?;
    println!("Done.");

    // Main Benchmark Loop
    let mut prefill_results = Vec::new();
    let mut decode_results = Vec::new();

    let pb = ProgressBar::new(args.iterations as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} trials completed",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    for _ in 0..args.iterations {
        match run_trial(
            &client,
            &args.model,
            prompt,
            args.ctx,
            args.num_predict,
            args.temperature,
            &args.host,
        )
        .await
        {
            Ok(metrics) => {
                prefill_results.push(metrics.prefill_tps);
                decode_results.push(metrics.decode_tps);
            }
            Err(e) => eprintln!("\n❌ Trial failed: {}", e),
        }
        pb.inc(1);
    }
    pb.finish_and_clear();

    // Statistical Analysis
    if prefill_results.is_empty() {
        println!("All trials failed. Verify Ollama is running.");
        return Ok(());
    }

    let avg_prefill: f64 = prefill_results.iter().sum::<f64>() / prefill_results.len() as f64;
    let avg_decode: f64 = decode_results.iter().sum::<f64>() / decode_results.len() as f64;

    let max_prefill = prefill_results.iter().cloned().fold(f64::NAN, f64::max);
    let min_prefill = prefill_results.iter().cloned().fold(f64::NAN, f64::min);
    let max_decode = decode_results.iter().cloned().fold(f64::NAN, f64::max);
    let min_decode = decode_results.iter().cloned().fold(f64::NAN, f64::min);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  📊  Benchmark Results Summary                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Prompt Processing (Prefill)                                ║");
    println!(
        "║    Average: {:>10.2}  Min: {:>10.2}  Max: {:>10.2}  tokens/sec",
        avg_prefill, min_prefill, max_prefill
    );
    println!("║  Token Generation (Decode)                                  ║");
    println!(
        "║    Average: {:>10.2}  Min: {:>10.2}  Max: {:>10.2}  tokens/sec",
        avg_decode, min_decode, max_decode
    );
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}
