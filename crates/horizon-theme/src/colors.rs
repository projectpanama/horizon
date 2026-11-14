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
