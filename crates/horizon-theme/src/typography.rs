use std::collections::BTreeMap;

/// OpenType feature tags and their values. Zero explicitly disables a feature.
pub type FontFeatures = BTreeMap<String, u16>;

/// A semantic typography role. Themes change its values, not its identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum TextRole {
    Display,
    Title1,
    Title2,
    Title3,
    Headline,
    Body,
    Callout,
    Label,
    Caption,
    Mono,
}

impl TextRole {
    /// All supported semantic text roles in their serialized order.
    pub const ALL: [Self; 10] = [
        Self::Display,
        Self::Title1,
        Self::Title2,
        Self::Title3,
        Self::Headline,
        Self::Body,
        Self::Callout,
        Self::Label,
        Self::Caption,
        Self::Mono,
    ];

    /// The corresponding key in `[typography.style]`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Display => "display",
            Self::Title1 => "title-1",
            Self::Title2 => "title-2",
            Self::Title3 => "title-3",
            Self::Headline => "headline",
            Self::Body => "body",
            Self::Callout => "callout",
            Self::Label => "label",
            Self::Caption => "caption",
            Self::Mono => "mono",
        }
    }
}

/// The slant applied to a font.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum FontSlant {
    #[default]
    Normal,
    Italic,
}

/// An ordered fallback stack and its font-specific features.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FontDefinition {
    pub families: Vec<String>,
    pub features: FontFeatures,
}

impl FontDefinition {
    /// Creates a font definition with no feature overrides.
    pub fn new(families: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            families: families.into_iter().map(Into::into).collect(),
            features: FontFeatures::new(),
        }
    }

    /// Validates that the fallback stack and feature tags are usable.
    pub fn validate(&self) -> Result<(), String> {
        if self.families.is_empty() || self.families.iter().any(|name| name.trim().is_empty()) {
            return Err("font families must be nonempty".into());
        }

        validate_features(&self.features)
    }
}

/// A complete, unscaled text style. `features` override the font's features by tag.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeStyle {
    pub font: FontDefinition,
    pub size: f32,
    pub line_height: f32,
    pub weight: f32,
    pub slant: FontSlant,
    pub tracking: f32,
    pub features: FontFeatures,
}

impl TypeStyle {
    /// Validates the font, dimensions, weight, tracking, and feature tags.
    pub fn validate(&self) -> Result<(), String> {
        self.font.validate()?;

        for (name, value) in [("size", self.size), ("line-height", self.line_height)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and positive"));
            }
        }
        if !self.weight.is_finite() || !(1.0..=1000.0).contains(&self.weight) {
            return Err("weight must be between 1 and 1000".into());
        }
        if !self.tracking.is_finite() {
            return Err("tracking must be finite".into());
        }

        validate_features(&self.features)
    }
}

/// A complete typography system, including font definitions and semantic styles.
#[derive(Debug, Clone, PartialEq)]
pub struct Typography {
    fonts: BTreeMap<String, FontDefinition>,
    styles: [TypeStyle; 10],
}

impl Typography {
    /// Returns the compiled style assigned to `role`.
    pub fn style(&self, role: TextRole) -> &TypeStyle {
        &self.styles[role as usize]
    }

    /// Returns a compiled font profile by name.
    pub fn font(&self, name: &str) -> Option<&FontDefinition> {
        self.fonts.get(name)
    }

    /// Replaces one role for programmatic themes. Invalid styles leave it unchanged.
    pub fn set_style(&mut self, role: TextRole, style: TypeStyle) -> Result<(), String> {
        style.validate()?;
        self.styles[role as usize] = style;
        Ok(())
    }

    /// Builds the typography from tokens (the nested `[typography]` TOML table).
    pub(crate) fn from_tokens(root: &toml::Table) -> Result<Self, String> {
        check_keys(root, &["font", "style"])?;

        let mut fonts = BTreeMap::new();
        for (name, value) in table(root, "font")? {
            let font_table = value
                .as_table()
                .ok_or_else(|| error(format!("font.{name} must be a table")))?;
            check_keys(font_table, &["families", "features"])?;
            let families = font_table
                .get("families")
                .and_then(toml::Value::as_array)
                .ok_or_else(|| error(format!("font.{name}.families must be an array")))?
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| error(format!("font.{name}.families must contain strings")))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let definition = FontDefinition {
                families,
                features: features(font_table)?,
            };
            definition
                .validate()
                .map_err(|validation_error| error(format!("font.{name}: {validation_error}")))?;
            fonts.insert(name.clone(), definition);
        }

        let style_table = table(root, "style")?;
        check_keys(style_table, &TextRole::ALL.map(TextRole::name))?;

        let mut styles = Vec::new();
        for role in TextRole::ALL {
            let parsed = (|| {
                let style = table(style_table, role.name())?;
                check_keys(
                    style,
                    &[
                        "font",
                        "size",
                        "line-height",
                        "weight",
                        "slant",
                        "tracking",
                        "features",
                    ],
                )?;
                let font_name = string(style, "font")?;
                let font = fonts
                    .get(font_name)
                    .ok_or_else(|| error(format!("unknown font profile {font_name:?}")))?
                    .clone();
                let slant = match string(style, "slant")? {
                    "normal" => FontSlant::Normal,
                    "italic" => FontSlant::Italic,
                    other => return Err(error(format!("unknown slant {other:?}"))),
                };
                let style = TypeStyle {
                    font,
                    slant,
                    size: number(style, "size")?,
                    line_height: number(style, "line-height")?,
                    weight: number(style, "weight")?,
                    tracking: number(style, "tracking")?,
                    features: features(style)?,
                };
                style.validate()?;
                Ok(style)
            })()
            .map_err(|parse_error: String| {
                error(format!("style.{}: {parse_error}", role.name()))
            })?;
            styles.push(parsed);
        }

        Ok(Self {
            fonts,
            styles: styles.try_into().expect("all ten roles were compiled"),
        })
    }
}

fn error(message: impl Into<String>) -> String {
    message.into()
}

fn check_keys(table: &toml::Table, allowed: &[&str]) -> Result<(), String> {
    for key in table.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(error(format!("unknown typography key {key:?}")));
        }
    }
    Ok(())
}

fn table<'a>(table: &'a toml::Table, key: &str) -> Result<&'a toml::Table, String> {
    table
        .get(key)
        .and_then(toml::Value::as_table)
        .ok_or_else(|| error(format!("missing or invalid table {key}")))
}

fn string<'a>(table: &'a toml::Table, key: &str) -> Result<&'a str, String> {
    table
        .get(key)
        .and_then(toml::Value::as_str)
        .ok_or_else(|| error(format!("missing or invalid string {key}")))
}

fn number(table: &toml::Table, key: &str) -> Result<f32, String> {
    let value = table
        .get(key)
        .and_then(|value| {
            value
                .as_float()
                .or_else(|| value.as_integer().map(|integer| integer as f64))
        })
        .ok_or_else(|| error(format!("missing or invalid number {key}")))?;
    Ok(value as f32)
}

fn features(table: &toml::Table) -> Result<FontFeatures, String> {
    let Some(value) = table.get("features") else {
        return Ok(FontFeatures::new());
    };
    let feature_table = value
        .as_table()
        .ok_or_else(|| error("features must be a tag/value table"))?;
    let features = feature_table
        .iter()
        .map(|(tag, feature_value)| {
            let feature_value = feature_value
                .as_integer()
                .and_then(|integer| u16::try_from(integer).ok())
                .ok_or_else(|| {
                    error(format!(
                        "feature {tag:?} must be an unsigned 16-bit integer"
                    ))
                })?;
            Ok((tag.clone(), feature_value))
        })
        .collect::<Result<FontFeatures, String>>()?;
    validate_features(&features)?;
    Ok(features)
}

fn validate_features(features: &FontFeatures) -> Result<(), String> {
    for tag in features.keys() {
        if tag.len() != 4 || !tag.bytes().all(|byte| (0x20..=0x7e).contains(&byte)) {
            return Err(error(format!(
                "invalid OpenType feature tag {tag:?}; expected four printable ASCII characters"
            )));
        }
    }
    Ok(())
}
