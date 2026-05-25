use reqwest::Client;
use serde::Serialize;

use crate::types::{ModelDetails, ModelInfo, OllamaTags};

/// Fetches the list of models from the Ollama API.
pub async fn fetch_tags(client: &Client, host: &str) -> Option<OllamaTags> {
    client
        .get(format!("{}/api/tags", host))
        .send()
        .await
        .ok()?
        .json::<OllamaTags>()
        .await
        .ok()
}

/// Extracted model metadata for display.
#[derive(Debug, Clone, Serialize)]
pub struct ModelDetailsPlain {
    pub params: String,
    pub quant: String,
    pub family: String,
    pub size: String,
}

impl ModelDetailsPlain {
    pub fn from_model_info(mi: &ModelInfo) -> Self {
        Self {
            params: mi
                .details
                .parameter_size
                .clone()
                .unwrap_or_else(|| "N/A".to_string()),
            quant: mi
                .details
                .quantization_level
                .clone()
                .unwrap_or_else(|| "N/A".to_string()),
            family: Self::format_family(&mi.details),
            size: format!("{:.1} GB", mi.size as f64 / 1_073_741_824.0),
        }
    }

    pub fn na() -> Self {
        Self {
            params: "N/A".to_string(),
            quant: "N/A".to_string(),
            family: "N/A".to_string(),
            size: "N/A".to_string(),
        }
    }

    fn format_family(details: &ModelDetails) -> String {
        details
            .families
            .as_ref()
            .filter(|v| !v.is_empty())
            .map(|v| v.join(", "))
            .or_else(|| details.family.clone())
            .unwrap_or_else(|| "N/A".to_string())
    }
}

/// Computes statistics for a slice of f64 values.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Stats {
    pub avg: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
}

impl Stats {
    pub fn compute(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                avg: 0.0,
                min: 0.0,
                max: 0.0,
                stddev: 0.0,
            };
        }
        let len = values.len() as f64;
        let avg = values.iter().sum::<f64>() / len;
        let min = values.iter().cloned().fold(f64::NAN, f64::min);
        let max = values.iter().cloned().fold(f64::NAN, f64::max);
        let stddev = if values.len() < 2 {
            0.0
        } else {
            let variance = values.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / (len - 1.0);
            variance.sqrt()
        };
        Self {
            avg,
            min,
            max,
            stddev,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelDetails;

    // ── Stats tests ────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let s = Stats::compute(&[]);
        assert_eq!(s.avg, 0.0);
        assert_eq!(s.min, 0.0);
        assert_eq!(s.max, 0.0);
        assert_eq!(s.stddev, 0.0);
    }

    #[test]
    fn test_stats_single_value() {
        let s = Stats::compute(&[42.0]);
        assert_eq!(s.avg, 42.0);
        assert_eq!(s.min, 42.0);
        assert_eq!(s.max, 42.0);
        assert_eq!(s.stddev, 0.0);
    }

    #[test]
    fn test_stats_two_values() {
        let s = Stats::compute(&[10.0, 20.0]);
        assert!((s.avg - 15.0).abs() < 1e-9);
        assert_eq!(s.min, 10.0);
        assert_eq!(s.max, 20.0);
        // stddev of [10, 20] = sqrt(50) ≈ 7.071...
        assert!((s.stddev - 7.071_067_811_865_475).abs() < 1e-9);
    }

    #[test]
    fn test_stats_known() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = Stats::compute(&data);
        assert!((s.avg - 5.0).abs() < 1e-9);
        assert_eq!(s.min, 2.0);
        assert_eq!(s.max, 9.0);
        // population sample stddev ≈ 2.138
        assert!((s.stddev - 2.138_089_935).abs() < 1e-6);
    }

    // ── ModelDetailsPlain tests ────────────────────────────────────────────

    fn make_model_info(
        name: &str,
        size: u64,
        param_size: Option<&str>,
        quant: Option<&str>,
        family: Option<&str>,
        families: Option<Vec<String>>,
    ) -> ModelInfo {
        ModelInfo {
            name: name.to_string(),
            size,
            details: ModelDetails {
                format: None,
                family: family.map(|s| s.to_string()),
                families,
                parameter_size: param_size.map(|s| s.to_string()),
                quantization_level: quant.map(|s| s.to_string()),
            },
        }
    }

    #[test]
    fn test_model_details_from_info() {
        let mi = make_model_info(
            "test:7b",
            8_000_000_000,
            Some("7B"),
            Some("Q4_K_M"),
            Some("llama"),
            None,
        );
        let d = ModelDetailsPlain::from_model_info(&mi);
        assert_eq!(d.params, "7B");
        assert_eq!(d.quant, "Q4_K_M");
        assert_eq!(d.family, "llama");
        assert_eq!(d.size, "7.5 GB");
    }

    #[test]
    fn test_model_details_na() {
        let d = ModelDetailsPlain::na();
        assert_eq!(d.params, "N/A");
        assert_eq!(d.quant, "N/A");
        assert_eq!(d.family, "N/A");
        assert_eq!(d.size, "N/A");
    }

    #[test]
    fn test_model_details_families_join() {
        let mi = make_model_info(
            "test:7b",
            4_000_000_000,
            None,
            None,
            None,
            Some(vec!["llama".into(), "moe".into()]),
        );
        let d = ModelDetailsPlain::from_model_info(&mi);
        assert_eq!(d.family, "llama, moe");
    }

    #[test]
    fn test_model_details_empty_families_fallback() {
        let mi = make_model_info(
            "test:7b",
            4_000_000_000,
            None,
            None,
            Some("mistral".into()),
            Some(vec![]),
        );
        let d = ModelDetailsPlain::from_model_info(&mi);
        assert_eq!(d.family, "mistral");
    }

    #[test]
    fn test_model_size_gb() {
        // 1 GB = 1_073_741_824 bytes
        let mi = make_model_info("x", 1_073_741_824, None, None, None, None);
        let d = ModelDetailsPlain::from_model_info(&mi);
        assert_eq!(d.size, "1.0 GB");

        let mi = make_model_info("x", 5_368_709_120, None, None, None, None);
        let d = ModelDetailsPlain::from_model_info(&mi);
        assert_eq!(d.size, "5.0 GB");
    }
}
