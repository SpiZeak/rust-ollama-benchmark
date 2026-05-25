# Ollama Rust Hardware Benchmark Suite

A command-line tool that measures how fast your hardware runs [Ollama](https://ollama.com) language models. It reports **prefill speed** (how quickly the model processes your prompt) and **decode speed** (how quickly it generates tokens), expressed in tokens per second.

## Prerequisites

### 1. Install Rust

If you don't have Rust installed, run:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then follow the on-screen instructions and restart your terminal (or run `source ~/.cargo/env`).

### 2. Install and start Ollama

Download Ollama from [ollama.com/download](https://ollama.com/download) and make sure it's running:

```bash
ollama serve
```

### 3. Pull a model

You need at least one model downloaded before benchmarking. For example:

```bash
ollama pull qwen3.5:9b-q4_K_M
```

## Building

Clone this repository and build the release binary (this takes a minute the first time):

```bash
git clone https://github.com/SpiZeak/rust-ollama-benchmark
cd rust-ollama-benchmark
cargo build --release
```

The compiled binary will be at `./target/release/ollama-bench`.

## Usage

### Benchmark a single model (default)

```bash
./target/release/ollama-bench
```

This benchmarks `qwen3.5:9b-q4_K_M` with 5 iterations and prints a results summary.

### Specify a different model

```bash
./target/release/ollama-bench --model llama3.2:3b
```

### Compare multiple models side by side

```bash
./target/release/ollama-bench --compare llama3.2:3b qwen3.5:9b-q4_K_M mistral:7b
```

### Run more iterations for a more stable average

```bash
./target/release/ollama-bench --iterations 10
```

### Use a custom prompt

```bash
./target/release/ollama-bench --prompt "Explain the theory of relativity in simple terms."
```

### Output results as JSON

```bash
./target/release/ollama-bench --json
```

Useful for programmatic consumption or saving to a file:

```bash
./target/release/ollama-bench --json > results.json
./target/release/ollama-bench --compare llama3.2:3b mistral:7b --json > comparison.json
```

### Connect to a remote Ollama instance

```bash
./target/release/ollama-bench --host http://192.168.1.100:11434
```

## All Options

| Flag            | Short | Default                  | Description                                             |
| --------------- | ----- | ------------------------ | ------------------------------------------------------- |
| `--model`       | `-m`  | `qwen3.5:9b-q4_K_M`      | Model to benchmark (single mode)                        |
| `--compare`     | `-C`  | —                        | Two or more models to compare side by side              |
| `--iterations`  | `-i`  | `5`                      | Number of benchmark runs (a warmup run is always added) |
| `--num-predict` | —     | `256`                    | Maximum tokens to generate per run                      |
| `--ctx`         | `-c`  | `24576`                  | Context window size in tokens                           |
| `--temperature` | `-t`  | `0.2`                    | Sampling temperature (lower = more deterministic)       |
| `--prompt`      | —     | _(built-in)_             | Custom prompt to use for all runs                       |
| `--host`        | —     | `http://localhost:11434` | Ollama API base URL                                     |
| `--json`        | —     | `false`                  | Output results as JSON instead of pretty tables         |

## Understanding the Output

```
║  Prompt Processing (Prefill)
║    Average:      412.50  Min:      398.10  Max:      430.20  tokens/sec
║  Token Generation (Decode)
║    Average:       38.74  Min:       37.90  Max:       39.50  tokens/sec
```

- **Prefill** — how fast the model reads and processes your input prompt. Higher is better.
- **Decode** — how fast the model generates output tokens one by one. Higher is better. This is the number most people experience as "response speed".
- **Average / Min / Max / Std** — statistics across all iterations. A narrow min–max range and low standard deviation means consistent performance.

## JSON Output Format

When `--json` is used, the tool outputs a JSON object with the following structure:

```json
{
  "system": {
    "os": "Linux 6.8.0-52-generic x86_64",
    "cpu": "AMD Ryzen 9 7950X",
    "ram_total": "64.0 GiB",
    "gpu": "NVIDIA: GeForce RTX 4090",
    "ollama_version": "0.6.0",
    "device": "GPU",
    "model_name": "qwen3.5:9b-q4_K_M",
    "model_params": "9B",
    "model_quant": "Q4_K_M",
    "model_family": "qwen3.5",
    "model_size": "5.5 GB",
    "kv_cache_type": "q4_K_M",
    "ctx": 24576,
    "iterations": 5
  },
  "results": [
    {
      "model": "qwen3.5:9b-q4_K_M",
      "avg_prefill": 412.5,
      "min_prefill": 398.1,
      "max_prefill": 430.2,
      "stddev_prefill": 11.3,
      "avg_decode": 38.74,
      "min_decode": 37.9,
      "max_decode": 39.5,
      "stddev_decode": 0.58,
      "params": "9B",
      "quant": "Q4_K_M",
      "family": "qwen3.5",
      "size": "5.5 GB"
    }
  ]
}
```

## Tips

- For more reliable numbers, close other GPU-intensive applications before running.
- Use `--iterations 10` or higher for a more stable average.
- Decode speed below ~10 tokens/sec will feel slow in interactive use; above ~30 tokens/sec feels fast.
- If you get a connection error, make sure `ollama serve` is running and the `--host` URL is correct.

## Development

Run tests:

```bash
cargo test
```

Run clippy for lint checks:

```bash
cargo clippy
```
