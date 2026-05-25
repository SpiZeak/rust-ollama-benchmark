mod benchmark;
mod cli;
mod display;
mod system;
mod types;
mod utils;

use clap::Parser;
use reqwest::Client;

use benchmark::{benchmark_model, run_benchmark_iterations, run_trial};
use cli::Args;
use display::{
    print_comparison_header, print_comparison_table, print_model_card, print_single_results,
    print_system_info,
};
use system::gather_system_info;
use types::{BenchConfig, BenchmarkOutput, DEFAULT_PROMPT, WARMUP_PROMPT};
use utils::Stats;

/// Run in single-model benchmark mode.
async fn run_single(
    client: &Client,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = args.prompt.as_deref().unwrap_or(DEFAULT_PROMPT);

    // Gather and display system info
    let sys = gather_system_info(client, &args.host, &args.model, args.ctx, args.iterations).await;
    print_system_info(
        &sys.os,
        &sys.cpu,
        &sys.ram_total,
        &sys.gpu,
        &sys.device,
        args.ctx,
        args.iterations,
    );
    print_model_card(
        &sys.ollama_version,
        &sys.model_name,
        &sys.model_params,
        &sys.model_quant,
        &sys.model_family,
        &sys.model_size,
        &sys.kv_cache_type,
    );

    let config = BenchConfig::from_args(args, prompt);

    // Warmup
    let warmup_config = BenchConfig {
        prompt: WARMUP_PROMPT.to_string(),
        ..config.clone()
    };
    print!("⏳ Priming {} (Warmup run)... ", sys.device);
    let _ = run_trial(client, &warmup_config).await?;
    println!("Done.");

    // Benchmark
    let (prefill_results, decode_results) = run_benchmark_iterations(client, &config).await;

    if decode_results.is_empty() {
        println!("All trials failed. Verify Ollama is running.");
        return Ok(());
    }

    let prefill_stats = Stats::compute(&prefill_results);
    let decode_stats = Stats::compute(&decode_results);

    // JSON output
    if args.json {
        let sys_json = types::SystemInfoJson::from(&sys);
        let result = types::ModelBenchmarkResult {
            model: args.model.to_string(),
            avg_prefill: prefill_stats.avg,
            min_prefill: prefill_stats.min,
            max_prefill: prefill_stats.max,
            stddev_prefill: prefill_stats.stddev,
            avg_decode: decode_stats.avg,
            min_decode: decode_stats.min,
            max_decode: decode_stats.max,
            stddev_decode: decode_stats.stddev,
            params: sys.model_params.clone(),
            quant: sys.model_quant.clone(),
            family: sys.model_family.clone(),
            size: sys.model_size.clone(),
        };
        let output = BenchmarkOutput {
            system: sys_json,
            results: vec![result],
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
        return Ok(());
    }

    // Pretty-print results
    print_single_results(&prefill_stats, &decode_stats, prefill_results.is_empty());
    Ok(())
}

/// Run in comparison (multi-model) benchmark mode.
async fn run_comparison(
    client: &Client,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = args.prompt.as_deref().unwrap_or(DEFAULT_PROMPT);

    print_comparison_header(&args.compare);

    let base_config = BenchConfig::from_args(args, prompt);

    let mut results: Vec<types::ModelBenchmarkResult> = Vec::new();
    for model in &args.compare {
        print!("  ▶  Benchmarking {} ... ", model);
        let model_config = base_config.for_model(model);
        match benchmark_model(client, &model_config).await {
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

    // JSON output
    if args.json {
        let output = BenchmarkOutput {
            system: types::SystemInfoJson {
                os: String::new(),
                cpu: String::new(),
                ram_total: String::new(),
                gpu: String::new(),
                ollama_version: String::new(),
                device: String::new(),
                model_name: String::new(),
                model_params: String::new(),
                model_quant: String::new(),
                model_family: String::new(),
                model_size: String::new(),
                kv_cache_type: String::new(),
                ctx: args.ctx,
                iterations: args.iterations,
            },
            results,
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
        return Ok(());
    }

    // Pretty-print
    print_comparison_table(&results);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if !args.compare.is_empty() {
        return run_comparison(&Client::new(), &args).await;
    }

    run_single(&Client::new(), &args).await
}
