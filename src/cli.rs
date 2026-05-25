use clap::Parser;

use crate::types::DEFAULT_HOST;
use crate::types::DEFAULT_MODEL;

#[derive(Parser, Debug)]
#[command(author, version, about = "Ollama Rust Hardware Benchmark Suite")]
pub struct Args {
    /// Model to benchmark (single mode)
    #[arg(short, long, default_value_t = std::borrow::Cow::Borrowed(DEFAULT_MODEL))]
    pub model: std::borrow::Cow<'static, str>,

    /// Models to compare (comparison mode; pass 2+ models)
    #[arg(short = 'C', long, num_args = 2..)]
    pub compare: Vec<String>,

    /// Number of benchmark iterations
    #[arg(short, long, default_value_t = 5)]
    pub iterations: usize,

    /// Max tokens to generate per trial (limits decode time)
    #[arg(long, default_value_t = 256)]
    pub num_predict: u32,

    /// Context window size in tokens
    #[arg(short, long, default_value_t = 24576)]
    pub ctx: u32,

    /// Temperature for generation
    #[arg(short, long, default_value_t = 0.2)]
    pub temperature: f32,

    /// Custom prompt (uses default if omitted)
    #[arg(long)]
    pub prompt: Option<String>,

    /// Ollama API host URL
    #[arg(long, default_value_t = std::borrow::Cow::Borrowed(DEFAULT_HOST))]
    pub host: std::borrow::Cow<'static, str>,
}
