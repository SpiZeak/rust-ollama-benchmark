use crate::types::ModelBenchmarkResult;
use crate::utils::Stats;

// ─── Box-drawing constants ──────────────────────────────────────────────────

const SINGLE_W: usize = 86;
const COMPARE_W: usize = 90;

// ─── System info header ─────────────────────────────────────────────────────

pub fn print_system_info(
    os: &str,
    cpu: &str,
    ram: &str,
    gpu: &str,
    device: &str,
    ctx: u32,
    iterations: usize,
) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  🚀  Ollama Rust Hardware Benchmark Suite                   ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  System Info                                                ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  OS:             {}", os);
    println!("║  CPU:            {}", cpu);
    println!("║  RAM:            {}", ram);
    println!("║  GPU:            {}", gpu);
    println!("║  Inference:      {}", device);
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Benchmark Config                                           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Context Window:  {} tokens", ctx);
    println!("║  Iterations:      {} runs (+ 1 warmup)", iterations);
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}

// ─── Model detail card ──────────────────────────────────────────────────────

pub fn print_model_card(
    ollama_version: &str,
    model_name: &str,
    params: &str,
    quant: &str,
    family: &str,
    size: &str,
    kv_cache: &str,
) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Ollama & Model                                             ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Ollama Version: {}", ollama_version);
    println!("║  Model:          {}", model_name);
    println!("║  Parameters:     {}", params);
    println!("║  Quantization:   {}", quant);
    println!("║  Architecture:   {}", family);
    println!("║  Model Size:     {}", size);
    println!("║  KV Cache Type:  {}", kv_cache);
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}

// ─── Comparison mode ────────────────────────────────────────────────────────

pub fn print_comparison_header(models: &[String]) {
    println!(
        "🔍  Comparison mode: benchmarking {} models\n",
        models.len()
    );
}

pub fn print_comparison_table(results: &[ModelBenchmarkResult]) {
    if results.is_empty() {
        return;
    }

    println!("\n╔{:═^COMPARE_W$}╗", "");
    println!("║  📊  Model Comparison Results                                  ║");
    println!("╠{:═^COMPARE_W$}╣", "");

    // Header row
    println!(
        "║  {:<23}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8} ║",
        "Model", "Params", "AvgPre", "StdPre", "AvgDec", "StdDec", "MinDec"
    );
    println!(
        "╠{:─^26}╬{:─^10}╬{:─^10}╬{:─^10}╬{:─^10}╬{:─^10}╬{:─^10}╣",
        "", "", "", "", "", "", ""
    );

    for r in results {
        println!(
            "║  {:<23}  {:>8}  {:>8.1}  {:>8.1}  {:>8.1}  {:>8.1}  {:>8.1} ║",
            r.model,
            r.params,
            r.avg_prefill,
            r.stddev_prefill,
            r.avg_decode,
            r.stddev_decode,
            r.min_decode
        );
    }

    println!("╚{:═^COMPARE_W$}╝", "");

    // Per-model detail cards
    for r in results {
        println!("\n╔{:═^SINGLE_W$}╗", "");
        println!("║  📋  {:<75} ║", r.model);
        println!("╠{:═^SINGLE_W$}╣", "");
        println!("║  Architecture:   {:<65} ║", r.family);
        println!("║  Parameters:     {:<65} ║", r.params);
        println!("║  Quantization:   {:<65} ║", r.quant);
        println!("║  Model Size:     {:<65} ║", r.size);
        println!("╠{:═^SINGLE_W$}╣", "");
        println!(
            "║  Prefill (t/s)   Avg: {:>10.2}  Min: {:>10.2}  Max: {:>10.2}  Std: {:>10.2}  ║",
            r.avg_prefill, r.min_prefill, r.max_prefill, r.stddev_prefill
        );
        println!(
            "║  Decode  (t/s)   Avg: {:>10.2}  Min: {:>10.2}  Max: {:>10.2}  Std: {:>10.2}  ║",
            r.avg_decode, r.min_decode, r.max_decode, r.stddev_decode
        );
        println!("╚{:═^SINGLE_W$}╝", "");
    }
}

// ─── Single mode results ────────────────────────────────────────────────────

/// `prefill_empty` is `true` when all runs were KV-cache hits (no prefill data).
pub fn print_single_results(prefill_stats: &Stats, decode_stats: &Stats, prefill_empty: bool) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  📊  Benchmark Results Summary                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Prompt Processing (Prefill)                                ║");
    if prefill_empty {
        println!("║    (all runs were KV-cache hits — prefill not measured)     ║");
    } else {
        println!(
            "║    Avg: {:>10.2}  Min: {:>10.2}  Max: {:>10.2}  Std: {:>10.2}  t/s",
            prefill_stats.avg, prefill_stats.min, prefill_stats.max, prefill_stats.stddev
        );
    }
    println!("║  Token Generation (Decode)                                  ║");
    println!(
        "║    Avg: {:>10.2}  Min: {:>10.2}  Max: {:>10.2}  Std: {:>10.2}  t/s",
        decode_stats.avg, decode_stats.min, decode_stats.max, decode_stats.stddev
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
}
