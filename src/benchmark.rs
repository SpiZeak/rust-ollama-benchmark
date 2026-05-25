use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;

use crate::types::{
    ModelBenchmarkResult, ModelInfo, OllamaOptions, OllamaRequest, OllamaResponse, OllamaTags,
    RunMetrics,
};

pub fn stddev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

pub async fn run_trial(
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

    let prefill_tps = match (res.prompt_eval_count, res.prompt_eval_duration) {
        (Some(count), Some(dur)) if dur > 0 => Some(count as f64 / (dur as f64 / 1_000_000_000.0)),
        _ => None, // KV cache hit — prefill was skipped by Ollama
    };
    let decode_sec = res.eval_duration as f64 / 1_000_000_000.0;

    Ok(RunMetrics {
        prefill_tps,
        decode_tps: res.eval_count as f64 / decode_sec,
    })
}

/// Benchmark a single model and return aggregated stats.
pub async fn benchmark_model(
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
                if let Some(p) = metrics.prefill_tps {
                    prefill_results.push(p);
                }
                decode_results.push(metrics.decode_tps);
            }
            Err(e) => eprintln!("\n  ⚠️  Trial failed for {}: {}", model, e),
        }
        pb.inc(1);
    }
    pb.finish_and_clear();

    if decode_results.is_empty() {
        println!("  ❌ All trials failed for {}. Skipping.", model);
        return None;
    }

    Some(ModelBenchmarkResult {
        model: model.to_string(),
        avg_prefill: prefill_results.iter().sum::<f64>() / prefill_results.len() as f64,
        min_prefill: prefill_results.iter().cloned().fold(f64::NAN, f64::min),
        max_prefill: prefill_results.iter().cloned().fold(f64::NAN, f64::max),
        stddev_prefill: stddev(&prefill_results),
        avg_decode: decode_results.iter().sum::<f64>() / decode_results.len() as f64,
        min_decode: decode_results.iter().cloned().fold(f64::NAN, f64::min),
        max_decode: decode_results.iter().cloned().fold(f64::NAN, f64::max),
        stddev_decode: stddev(&decode_results),
        params,
        quant,
        family,
        size,
    })
}
