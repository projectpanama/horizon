use crate::colors::Rgba;

const DEFAULT_STEPS: &[&str] = &[
    "50", "100", "200", "300", "400", "500", "600", "700", "800", "900",
];

const NEUTRAL_STEPS: &[&str] = &[
    "0", "50", "100", "150", "200", "300", "400", "500", "600", "700", "800", "850", "900", "950",
    "1000",
];

/// A collection of named colors used by the theme.
#[derive(Debug)]
pub(crate) struct Palette {
    neutral: Ramp<15>,
    accent: Ramp<10>,
    danger: Ramp<10>,
    warning: Ramp<10>,
    success: Ramp<10>,
}

impl Palette {
    /// Resolves a color reference to its corresponding RGBA value.
    pub(crate) fn color(&self, reference: &str) -> Result<Rgba, String> {
        let (ramp, step) = reference
            .split_once('.')
            .ok_or_else(|| format!("bad color reference {reference:?}"))?;

        match ramp {
            "neutral" => self.neutral.color(step),
            "accent" => self.accent.color(step),
            "danger" => self.danger.color(step),
            "warning" => self.warning.color(step),
            "success" => self.success.color(step),
            _ => Err(format!("unknown palette {ramp:?}")),
        }
    }

    /// Builds the palette from tokens (the nested `[palette]` TOML table).
    pub(crate) fn from_tokens(table: &toml::Table) -> Result<Self, String> {
        let parse = |name: &str, steps: &[&str]| -> Result<Vec<Rgba>, String> {
            let ramp = table
                .get(name)
                .and_then(toml::Value::as_table)
                .ok_or_else(|| format!("missing [palette.{name}]"))?;

            steps
                .iter()
                .map(|step| {
                    let hex = ramp
                        .get(*step)
                        .and_then(toml::Value::as_str)
                        .ok_or_else(|| format!("missing palette.{name}.{step}"))?;

                    Rgba::from_hex(hex)
                        .ok_or_else(|| format!("invalid hex in palette.{name}.{step}: {hex}"))
                })
                .collect()
        };

        // A closure cannot return arrays with different lengths, so it parses
        // into a Vec and this macro performs each typed array conversion.
        macro_rules! ramp {
            ($name:literal, $steps:expr, $length:literal) => {{
                let values: [Rgba; $length] = parse($name, $steps)?
                    .try_into()
                    .map_err(|_| format!("invalid number of steps in palette.{}", $name))?;

                Ramp {
                    name: $name,
                    values,
                    steps: $steps,
                }
            }};
        }

        Ok(Self {
            neutral: ramp!("neutral", NEUTRAL_STEPS, 15),
            accent: ramp!("accent", DEFAULT_STEPS, 10),
            danger: ramp!("danger", DEFAULT_STEPS, 10),
            warning: ramp!("warning", DEFAULT_STEPS, 10),
            success: ramp!("success", DEFAULT_STEPS, 10),
        })
    }
}

#[derive(Debug)]
struct Ramp<const N: usize> {
    name: &'static str,
    values: [Rgba; N],
    steps: &'static [&'static str],
}

impl<const N: usize> Ramp<N> {
    /// Resolves a color step to its corresponding RGBA value.
    fn color(&self, step: &str) -> Result<Rgba, String> {
        let index = self
            .steps
            .iter()
            .position(|candidate| *candidate == step)
            .ok_or_else(|| format!("unknown {} step {step:?}", self.name))?;

        self.values
            .get(index)
            .copied()
            .ok_or_else(|| format!("{} step {step:?} has no corresponding color", self.name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn palette_tokens() -> toml::Table {
        let mut palette = toml::Table::new();

        for (name, steps) in [
            ("neutral", NEUTRAL_STEPS),
            ("accent", DEFAULT_STEPS),
            ("danger", DEFAULT_STEPS),
            ("warning", DEFAULT_STEPS),
            ("success", DEFAULT_STEPS),
        ] {
            let mut ramp = toml::Table::new();
            for step in steps {
                ramp.insert((*step).to_string(), toml::Value::String("#010203".into()));
            }
            palette.insert(name.into(), toml::Value::Table(ramp));
        }

        palette
    }

    #[test]
    fn parses_neutral_and_color_ramps() {
        let palette = Palette::from_tokens(&palette_tokens()).unwrap();

        assert_eq!(
            palette.color("neutral.850").unwrap(),
            Rgba::new(1.0 / 255.0, 2.0 / 255.0, 3.0 / 255.0, 1.0)
        );
        assert_eq!(
            palette.color("accent.500").unwrap(),
            Rgba::new(1.0 / 255.0, 2.0 / 255.0, 3.0 / 255.0, 1.0)
        );
    }

    #[test]
    fn rejects_invalid_color_references() {
        let palette = Palette::from_tokens(&palette_tokens()).unwrap();

        assert_eq!(
            palette.color("neutral.999").unwrap_err(),
            "unknown neutral step \"999\""
        );
        assert_eq!(
            palette.color("unknown.500").unwrap_err(),
            "unknown palette \"unknown\""
        );
        assert_eq!(
            palette.color("neutral").unwrap_err(),
            "bad color reference \"neutral\""
        );
    }

    #[test]
    fn rejects_invalid_palette_values() {
        let mut map = palette_tokens();
        let neutral = map
            .get_mut("neutral")
            .and_then(toml::Value::as_table_mut)
            .unwrap();
        neutral.insert("850".into(), toml::Value::String("not-a-color".into()));

        assert_eq!(
            Palette::from_tokens(&map).unwrap_err(),
            "invalid hex in palette.neutral.850: not-a-color"
        );
    }
}
