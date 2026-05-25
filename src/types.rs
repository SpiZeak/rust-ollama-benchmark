use serde::{Deserialize, Serialize};

pub const DEFAULT_MODEL: &str = "qwen3.5:9b-q4_K_M";
pub const DEFAULT_HOST: &str = "http://localhost:11434";

// Simulate a heavy repository file analysis prompt
pub const DEFAULT_PROMPT: &str = "Analyze the architectural structure of this code block, check for any concurrency bottlenecks, and propose an idiomatic refactor.";

/// Short warmup prompt (avoids the heavy default prompt for warmup).
pub const WARMUP_PROMPT: &str = "Hello, world!";

// ─── Ollama API types ───────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
pub struct OllamaResponse {
    /// Absent when Ollama serves the prompt from the KV cache.
    #[serde(default)]
    pub prompt_eval_count: Option<u32>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>, // nanoseconds
    #[serde(default)]
    pub eval_count: Option<u32>,
    #[serde(default)]
    pub eval_duration: Option<u64>, // nanoseconds
}

#[derive(Serialize)]
pub struct OllamaOptions {
    pub num_ctx: u32,
    pub num_predict: u32,
    pub temperature: f32,
}

#[derive(Serialize)]
pub struct OllamaRequest<'a> {
    pub model: &'a str,
    pub prompt: &'a str,
    pub stream: bool,
    pub options: OllamaOptions,
}

#[derive(Deserialize, Debug)]
pub struct OllamaVersion {
    pub version: String,
}

#[derive(Deserialize, Debug)]
pub struct ModelDetails {
    #[allow(dead_code)]
    pub format: Option<String>,
    pub family: Option<String>,
    pub families: Option<Vec<String>>,
    pub parameter_size: Option<String>,
    pub quantization_level: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct ModelInfo {
    pub name: String,
    pub size: u64,
    pub details: ModelDetails,
}

#[derive(Deserialize, Debug)]
pub struct ProjectorInfo {
    pub cache_type: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct OllamaShow {
    #[allow(dead_code)]
    pub parameters: Option<String>,
    pub projectors: Option<Vec<ProjectorInfo>>,
}

#[derive(Deserialize, Debug)]
pub struct OllamaTags {
    pub models: Vec<ModelInfo>,
}

// ─── Application types ──────────────────────────────────────────────────────

pub struct SystemInfo {
    pub os: String,
    pub cpu: String,
    pub ram_total: String,
    pub gpu: String,
    pub ollama_version: String,
    pub model_name: String,
    pub model_params: String,
    pub model_quant: String,
    pub model_family: String,
    pub model_size: String,
    pub device: String, // "GPU" or "CPU"
    pub kv_cache_type: String,
    pub ctx: u32,
    pub iterations: usize,
}

pub struct RunMetrics {
    /// `None` when Ollama returned a KV-cache hit (no prefill was performed).
    pub prefill_tps: Option<f64>,
    pub decode_tps: f64,
}

/// Serializable benchmark result for one model.
#[derive(Debug, Clone, Serialize)]
pub struct ModelBenchmarkResult {
    pub model: String,
    pub avg_prefill: f64,
    pub min_prefill: f64,
    pub max_prefill: f64,
    pub stddev_prefill: f64,
    pub avg_decode: f64,
    pub min_decode: f64,
    pub max_decode: f64,
    pub stddev_decode: f64,
    pub params: String,
    pub quant: String,
    pub family: String,
    pub size: String,
}

/// Benchmark configuration shared across trials.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub model: String,
    pub prompt: String,
    pub ctx: u32,
    pub num_predict: u32,
    pub temperature: f32,
    pub host: String,
    pub iterations: usize,
}

impl BenchConfig {
    pub fn from_args(args: &crate::cli::Args, prompt: &str) -> Self {
        Self {
            model: args.model.to_string(),
            prompt: prompt.to_string(),
            ctx: args.ctx,
            num_predict: args.num_predict,
            temperature: args.temperature,
            host: args.host.to_string(),
            iterations: args.iterations,
        }
    }

    pub fn for_model(&self, model: &str) -> Self {
        let mut c = self.clone();
        c.model = model.to_string();
        c
    }
}

/// Top-level JSON output for the full benchmark run.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkOutput {
    pub system: SystemInfoJson,
    pub results: Vec<ModelBenchmarkResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemInfoJson {
    pub os: String,
    pub cpu: String,
    pub ram_total: String,
    pub gpu: String,
    pub ollama_version: String,
    pub device: String,
    pub model_name: String,
    pub model_params: String,
    pub model_quant: String,
    pub model_family: String,
    pub model_size: String,
    pub kv_cache_type: String,
    pub ctx: u32,
    pub iterations: usize,
}

impl From<&SystemInfo> for SystemInfoJson {
    fn from(s: &SystemInfo) -> Self {
        Self {
            os: s.os.clone(),
            cpu: s.cpu.clone(),
            ram_total: s.ram_total.clone(),
            gpu: s.gpu.clone(),
            ollama_version: s.ollama_version.clone(),
            device: s.device.clone(),
            model_name: s.model_name.clone(),
            model_params: s.model_params.clone(),
            model_quant: s.model_quant.clone(),
            model_family: s.model_family.clone(),
            model_size: s.model_size.clone(),
            kv_cache_type: s.kv_cache_type.clone(),
            ctx: s.ctx,
            iterations: s.iterations,
        }
    }
}
