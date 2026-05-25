use serde::{Deserialize, Serialize};

pub const DEFAULT_MODEL: &str = "qwen3.5:9b-q4_K_M";
pub const DEFAULT_HOST: &str = "http://localhost:11434";

// Simulate a heavy repository file analysis prompt
pub const DEFAULT_PROMPT: &str = "Analyze the architectural structure of this code block, check for any concurrency bottlenecks, and propose an idiomatic refactor.";

#[derive(Deserialize, Debug)]
pub struct OllamaResponse {
    // Absent when Ollama serves the prompt from the KV cache
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

pub struct SystemInfo {
    pub os: String,
    pub cpu: String,
    pub ram_total: String,
    pub gpu: String,
    pub ollama_version: String,
    pub model_params: String,
    pub model_quant: String,
    pub model_family: String,
    pub model_size: String,
    pub device: String, // "GPU" or "CPU"
    pub kv_cache_type: String,
}

pub struct RunMetrics {
    /// `None` when Ollama returned a KV-cache hit (no prefill was performed)
    pub prefill_tps: Option<f64>,
    pub decode_tps: f64,
}

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
