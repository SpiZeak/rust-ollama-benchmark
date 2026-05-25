use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;

use crate::types::{
    BenchConfig, ModelBenchmarkResult, OllamaOptions, OllamaRequest, OllamaResponse, RunMetrics,
};
use crate::utils::{fetch_tags, ModelDetailsPlain, Stats};

/// Run a single benchmark trial against the Ollama API.
pub async fn run_trial(
    client: &Client,
    config: &BenchConfig,
) -> Result<RunMetrics, Box<dyn std::error::Error>> {
    let payload = OllamaRequest {
        model: &config.model,
        prompt: &config.prompt,
        stream: false,
        options: OllamaOptions {
            num_ctx: config.ctx,
            num_predict: config.num_predict,
            temperature: config.temperature,
        },
    };

    let res: OllamaResponse = client
        .post(format!("{}/api/generate", config.host))
        .json(&payload)
        .send()
        .await?
        .json()
        .await?;

    let prefill_tps = match (res.prompt_eval_count, res.prompt_eval_duration) {
        (Some(count), Some(dur)) if dur > 0 => Some(count as f64 / (dur as f64 / 1_000_000_000.0)),
        _ => None,
    };
    let (eval_count, eval_duration) = match (res.eval_count, res.eval_duration) {
        (Some(c), Some(d)) if d > 0 => (c, d),
        _ => return Err("Ollama response missing eval_count or eval_duration".into()),
    };
    let decode_sec = eval_duration as f64 / 1_000_000_000.0;

    Ok(RunMetrics {
        prefill_tps,
        decode_tps: eval_count as f64 / decode_sec,
    })
}

/// Run one or more benchmark iterations (after warmup) and return per-run metrics.
pub async fn run_benchmark_iterations(
    client: &Client,
    config: &BenchConfig,
) -> (Vec<f64>, Vec<f64>) {
    let pb = ProgressBar::new(config.iterations as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:30.cyan/blue}] {pos}/{len} ({msg})",
        )
        .unwrap()
        .progress_chars("#>-"),
    );
    pb.set_message(config.model.clone());

    let mut prefill_results = Vec::new();
    let mut decode_results = Vec::new();

    for _ in 0..config.iterations {
        match run_trial(client, config).await {
            Ok(metrics) => {
                if let Some(p) = metrics.prefill_tps {
                    prefill_results.push(p);
                }
                decode_results.push(metrics.decode_tps);
            }
            Err(e) => eprintln!("\n  ⚠️  Trial failed for {}: {}", config.model, e),
        }
        pb.inc(1);
    }
    pb.finish_and_clear();

    (prefill_results, decode_results)
}

/// Benchmark a single model and return aggregated stats.
pub async fn benchmark_model(
    client: &Client,
    config: &BenchConfig,
) -> Option<ModelBenchmarkResult> {
    // Fetch model metadata
    let details: ModelDetailsPlain = match fetch_tags(client, &config.host).await {
        Some(tags) => tags
            .models
            .iter()
            .find(|m| {
                m.name == config.model
                    || m.name
                        .strip_suffix(":latest")
                        .map(|n| n == config.model.as_str())
                        .unwrap_or(false)
            })
            .map(ModelDetailsPlain::from_model_info)
            .unwrap_or_else(ModelDetailsPlain::na),
        None => ModelDetailsPlain::na(),
    };

    // Warmup with a short prompt
    let warmup_config = BenchConfig {
        prompt: crate::types::WARMUP_PROMPT.to_string(),
        ..config.clone()
    };
    let _ = run_trial(client, &warmup_config).await;

    let (prefill_results, decode_results) = run_benchmark_iterations(client, config).await;

    if decode_results.is_empty() {
        println!("  ❌ All trials failed for {}. Skipping.", config.model);
        return None;
    }

    let prefill_stats = Stats::compute(&prefill_results);
    let decode_stats = Stats::compute(&decode_results);

    Some(ModelBenchmarkResult {
        model: config.model.clone(),
        avg_prefill: prefill_stats.avg,
        min_prefill: prefill_stats.min,
        max_prefill: prefill_stats.max,
        stddev_prefill: prefill_stats.stddev,
        avg_decode: decode_stats.avg,
        min_decode: decode_stats.min,
        max_decode: decode_stats.max,
        stddev_decode: decode_stats.stddev,
        params: details.params,
        quant: details.quant,
        family: details.family,
        size: details.size,
    })
}
