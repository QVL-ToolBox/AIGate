//! Built-in, approximate price table for cost estimation.
//!
//! Prices are USD per 1M tokens and are a convenience estimate only — they go
//! stale and are not authoritative billing. Lookup is by `provider` + a model
//! **prefix**, so dated ids like `gpt-4o-mini-2024-07-18` match `gpt-4o-mini`.

use crate::types::Usage;

/// Input/output price, USD per 1M tokens.
#[derive(Debug, Clone, Copy)]
pub struct Pricing {
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
}

// (provider, model-prefix, input_per_mtok, output_per_mtok)
const PRICES: &[(&str, &str, f64, f64)] = &[
    ("openai", "gpt-4o-mini", 0.15, 0.60),
    ("openai", "gpt-4o", 2.50, 10.00),
    ("openai", "gpt-4.1", 2.00, 8.00),
    ("openai", "o4-mini", 1.10, 4.40),
    ("openai", "o3", 2.00, 8.00),
    ("mistral", "mistral-large", 2.00, 6.00),
    ("mistral", "mistral-small", 0.20, 0.60),
    ("mistral", "open-mistral-nemo", 0.15, 0.15),
    ("claude", "claude-opus-4", 15.00, 75.00),
    ("claude", "claude-sonnet-4", 3.00, 15.00),
    ("claude", "claude-haiku-4", 1.00, 5.00),
    ("claude", "claude-3-5-haiku", 0.80, 4.00),
    ("gemini", "gemini-2.5-pro", 1.25, 10.00),
    ("gemini", "gemini-2.5-flash", 0.30, 2.50),
    ("gemini", "gemini-2.0-flash", 0.10, 0.40),
    ("gemini", "gemini-1.5-pro", 1.25, 5.00),
];

/// Look up pricing for a provider + model (longest matching prefix wins).
pub fn pricing(provider: &str, model: &str) -> Option<Pricing> {
    PRICES
        .iter()
        .filter(|(p, m, _, _)| *p == provider && model.starts_with(m))
        // Prefer the most specific (longest) prefix match.
        .max_by_key(|(_, m, _, _)| m.len())
        .map(|(_, _, i, o)| Pricing {
            input_per_mtok: *i,
            output_per_mtok: *o,
        })
}

/// Estimate the USD cost of a usage record, or `None` for an unpriced model.
pub fn estimate_cost(provider: &str, model: &str, usage: &Usage) -> Option<f64> {
    let p = pricing(provider, model)?;
    Some(
        usage.prompt_tokens as f64 / 1e6 * p.input_per_mtok
            + usage.completion_tokens as f64 / 1e6 * p.output_per_mtok,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(prompt: u32, completion: u32) -> Usage {
        Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }

    #[test]
    fn exact_and_dated_models_match() {
        // 1M prompt @ 2.50 + 1M completion @ 10.00 = 12.50
        let cost = estimate_cost("openai", "gpt-4o", &usage(1_000_000, 1_000_000)).unwrap();
        assert!((cost - 12.50).abs() < 1e-9);
        // Dated id resolves via prefix to the same price.
        let dated = estimate_cost("openai", "gpt-4o-2024-08-06", &usage(1_000_000, 0)).unwrap();
        assert!((dated - 2.50).abs() < 1e-9);
    }

    #[test]
    fn most_specific_prefix_wins() {
        // "gpt-4o-mini" must not be priced as "gpt-4o".
        let mini = pricing("openai", "gpt-4o-mini").unwrap();
        assert!((mini.input_per_mtok - 0.15).abs() < 1e-9);
    }

    #[test]
    fn unknown_model_has_no_price() {
        assert!(estimate_cost("openai", "ghost-model", &usage(10, 10)).is_none());
        assert!(pricing("nope", "x").is_none());
    }
}
