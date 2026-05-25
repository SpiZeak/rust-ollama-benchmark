mod benchmark;
mod cli;
mod system;
mod types;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;

use benchmark::{benchmark_model, run_trial, stddev};
use cli::Args;
use system::gather_system_info;
use types::{DEFAULT_PROMPT, ModelBenchmarkResult};

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
            "║  📊  Model Comparison Results                                                        ║"
        );
        println!(
            "╠══════════════════════════════════════════════════════════════════════════════════════╣"
        );

        // Header
        println!(
            "║  {:<23}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8} ║",
            "Model", "Params", "AvgPre", "StdPre", "AvgDec", "StdDec", "MinDec"
        );
        println!(
            "╠{:─<26}╬{:─<9}╬{:─<9}╬{:─<9}╬{:─<9}╬{:─<9}╬{:─<9}╣",
            "", "", "", "", "", "", ""
        );

        for r in &results {
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

        println!(
            "╚══════════════════════════════════════════════════════════════════════════════════════╝"
        );

        // Per-model detail cards
        for r in &results {
            println!("\n╔════════════════════════════════════════════════════════════════╗");
            println!(
                "║  📋  {}                                             ║",
                r.model
            );
            println!("╠════════════════════════════════════════════════════════════════╣");
            println!("║  Architecture:   {}", r.family);
            println!("║  Parameters:     {}", r.params);
            println!("║  Quantization:   {}", r.quant);
            println!("║  Model Size:     {}", r.size);
            println!("╠════════════════════════════════════════════════════════════════╣");
            println!(
                "║  Prefill (tokens/s)   Avg: {:>10.2}  Min: {:>10.2}  Max: {:>10.2}  Std: {:>10.2}",
                r.avg_prefill, r.min_prefill, r.max_prefill, r.stddev_prefill
            );
            println!(
                "║  Decode  (tokens/s)   Avg: {:>10.2}  Min: {:>10.2}  Max: {:>10.2}  Std: {:>10.2}",
                r.avg_decode, r.min_decode, r.max_decode, r.stddev_decode
            );
            println!("╚════════════════════════════════════════════════════════════════╝");
        }

        return Ok(());
    }

    // --- Single model mode ---
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Ollama & Model                                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Ollama Version: {}", sys.ollama_version);
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
                if let Some(p) = metrics.prefill_tps {
                    prefill_results.push(p);
                }
                decode_results.push(metrics.decode_tps);
            }
            Err(e) => eprintln!("\n❌ Trial failed: {}", e),
        }
        pb.inc(1);
    }
    pb.finish_and_clear();

    // Statistical Analysis
    if decode_results.is_empty() {
        println!("All trials failed. Verify Ollama is running.");
        return Ok(());
    }

    let avg_decode: f64 = decode_results.iter().sum::<f64>() / decode_results.len() as f64;
    let max_decode = decode_results.iter().cloned().fold(f64::NAN, f64::max);
    let min_decode = decode_results.iter().cloned().fold(f64::NAN, f64::min);
    let std_decode = stddev(&decode_results);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  📊  Benchmark Results Summary                               ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Prompt Processing (Prefill)                                 ║");
    if prefill_results.is_empty() {
        println!("║    (all runs were KV-cache hits — prefill not measured)      ║");
    } else {
        let avg_prefill = prefill_results.iter().sum::<f64>() / prefill_results.len() as f64;
        let max_prefill = prefill_results.iter().cloned().fold(f64::NAN, f64::max);
        let min_prefill = prefill_results.iter().cloned().fold(f64::NAN, f64::min);
        let std_prefill = stddev(&prefill_results);
        println!(
            "║    Avg: {:>10.2}  Min: {:>10.2}  Max: {:>10.2}  Std: {:>10.2}  t/s",
            avg_prefill, min_prefill, max_prefill, std_prefill
        );
    }
    println!("║  Token Generation (Decode)                                   ║");
    println!(
        "║    Avg: {:>10.2}  Min: {:>10.2}  Max: {:>10.2}  Std: {:>10.2}  t/s",
        avg_decode, min_decode, max_decode, std_decode
    );
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}
