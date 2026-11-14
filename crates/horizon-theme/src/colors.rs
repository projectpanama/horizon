use crate::palette::Palette;

/// Selects the light or dark color set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorScheme {
    Light,
    Dark,
}

/// A collection of colors for a specific color scheme.
#[derive(Debug, Clone, PartialEq)]
pub struct Colors {
    /// Colors associated with surface components.
    pub surface: SurfaceColors,
}

impl Colors {
    /// Mixes two color sets together by a given amount.
    ///
    /// The `amount` parameter should be in the range `[0.0, 1.0]`, where `0.0`
    /// returns `self` and `1.0` returns `other`.
    pub fn mix(&self, other: &Self, amount: f32) -> Self {
        Self {
            surface: self.surface.mix(&other.surface, amount),
        }
    }

    /// Resolves a color scheme table using the supplied palette.
    ///
    /// Values may be palette references such as `"neutral.50"` or literal
    /// hexadecimal colors such as `"#ffffff"`.
    pub(crate) fn from_tokens(table: &toml::Table, palette: &Palette) -> Result<Self, String> {
        let parse = |name: &str| -> Result<Rgba, String> {
            let reference = table
                .get(name)
                .and_then(toml::Value::as_str)
                .ok_or_else(|| format!("missing color.{name}"))?;

            if reference.starts_with('#') {
                Rgba::from_hex(reference).ok_or_else(|| format!("invalid hex color {reference:?}"))
            } else {
                palette.color(reference)
            }
        };

        Ok(Self {
            surface: SurfaceColors {
                topbar: parse("surface.topbar")?,
            },
        })
    }
}

/// A color represented by normalized red, green, blue, and alpha channels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Rgba {
    /// Creates a color from normalized channel values.
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Parses a six-digit RGB hexadecimal color.
    ///
    /// Parsed colors always have an alpha value of `1.0`.
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.strip_prefix('#').unwrap_or(hex);
        if hex.len() != 6 || !hex.is_ascii() {
            return None;
        }

        let value = u32::from_str_radix(hex, 16).ok()?;
        let channel = |shift: u32| ((value >> shift) & 0xff) as f32 / 255.0;
        Some(Self::new(channel(16), channel(8), channel(0), 1.0))
    }

    /// Mixes two colors together by a given amount.
    ///
    /// The `amount` parameter should be in the range `[0.0, 1.0]`, where `0.0`
    /// returns `self` and `1.0` returns `other`.
    pub fn mix(&self, other: Self, amount: f32) -> Self {
        let amount = amount.clamp(0.0, 1.0);

        Self {
            r: lerp(self.r, other.r, amount),
            g: lerp(self.g, other.g, amount),
            b: lerp(self.b, other.b, amount),
            a: lerp(self.a, other.a, amount),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceColors {
    /// The color of the top bar.
    pub topbar: Rgba,
}

impl SurfaceColors {
    /// Mixes two surface color sets together by a given amount.
    fn mix(&self, other: &Self, amount: f32) -> Self {
        Self {
            topbar: self.topbar.mix(other.topbar, amount),
        }
    }
}

/// Linearly interpolates between two values by a given amount.
///
/// The `amount` parameter should be in the range `[0.0, 1.0]`, where `0.0`
/// returns `from` and `1.0` returns `to`.
fn lerp(from: f32, to: f32, amount: f32) -> f32 {
    from + (to - from) * amount
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hex_with_or_without_leading_hash() {
        let expected = Some(Rgba::new(
            0x1a as f32 / 255.0,
            0x2b as f32 / 255.0,
            0x3c as f32 / 255.0,
            1.0,
        ));

        assert_eq!(Rgba::from_hex("#1A2B3C"), expected);
        assert_eq!(Rgba::from_hex("1A2B3C"), expected);
    }

    #[test]
    fn rejects_invalid_hex_colors() {
        assert!(Rgba::from_hex("#12345").is_none());
        assert!(Rgba::from_hex("1234567").is_none());
        assert!(Rgba::from_hex("#GGHHII").is_none());
        assert!(Rgba::from_hex("").is_none());
    }
}
