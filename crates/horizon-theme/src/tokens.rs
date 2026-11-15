use std::path::{Path, PathBuf};

use crate::{ColorScheme, Colors, Typography, palette::Palette};

const BUILTIN: &str = include_str!("../assets/tokens.toml");
const TOKEN_RELATIVE_PATH: &str = "horizon/tokens.toml";

#[derive(Debug, Clone, PartialEq)]
pub struct Tokens {
    /// Colors used by the light color scheme.
    pub light: Colors,
    /// Colors used by the dark color scheme.
    pub dark: Colors,
    /// Typography shared by both color schemes.
    pub typography: Typography,
}

impl Tokens {
    /// Loads and compiles the built-in tokens with optional system and user layers.
    ///
    /// The layers are loaded in the following order:
    /// 1. Built-in tokens (always present)
    /// 2. `/usr/share/horizon/tokens.toml` (optional)
    /// 3. `$XDG_CONFIG_HOME/horizon/tokens.toml` or `$HOME/.config/horizon/tokens.toml` (optional)
    ///
    /// Each layer overrides matching values from earlier layers.
    ///
    /// If a layer is not found or cannot be read, it will be ignored with a warning.
    pub fn load() -> Result<Self, Error> {
        let mut warnings = Vec::new();

        let system = read_layer(
            &Path::new("/usr/share").join(TOKEN_RELATIVE_PATH),
            &mut warnings,
        );

        let user = xdg_path("XDG_CONFIG_HOME", ".config")
            .map(|config_dir| config_dir.join(TOKEN_RELATIVE_PATH))
            .as_deref()
            .and_then(|path| read_layer(path, &mut warnings));

        let layers: Vec<Layer> = [system, user].into_iter().flatten().collect();
        let (tokens, more_warnings) = Self::load_with(&layers)?;

        for warning in warnings.into_iter().chain(more_warnings) {
            eprintln!("horizon: {warning}");
        }

        Ok(tokens)
    }

    /// Returns the colors for the requested color scheme.
    pub fn colors(&self, scheme: ColorScheme) -> &Colors {
        match scheme {
            ColorScheme::Light => &self.light,
            ColorScheme::Dark => &self.dark,
        }
    }

    fn load_with(layers: &[Layer]) -> Result<(Self, Vec<String>), Error> {
        let base: toml::Table = BUILTIN.parse().map_err(|error| Error(format!("{error}")))?;

        let mut tokens = Self::from_table(&base)?;
        let mut merged = base.clone();
        let mut warnings = Vec::new();

        for layer in layers {
            let mut overlay: toml::Table = match layer.source.parse() {
                Ok(table) => table,
                Err(error) => {
                    warnings.push(format!("{}: {error}; ignoring it", layer.label));
                    continue;
                }
            };

            overlay.remove("meta");

            let mut candidate = merged.clone();

            merge(&mut candidate, overlay);

            match Self::from_table(&candidate) {
                Ok(compiled) => {
                    merged = candidate;
                    tokens = compiled;
                }
                Err(error) => {
                    warnings.push(format!("{}: {error}; ignoring it", layer.label));
                }
            }
        }

        Ok((tokens, warnings))
    }

    fn from_table(root: &toml::Table) -> Result<Self, Error> {
        // Traverses a TOML table using a path of keys.
        let table = |path: &[&str]| -> Result<&toml::Table, Error> {
            let mut current = root;
            for key in path {
                current = current
                    .get(*key)
                    .and_then(|value| value.as_table())
                    .ok_or_else(|| Error(format!("missing [{}]", path.join("."))))?;
            }
            Ok(current)
        };

        let palette = Palette::from_tokens(table(&["palette"])?).map_err(Error)?;

        Ok(Self {
            light: Colors::from_tokens(table(&["color", "light"])?, &palette).map_err(Error)?,
            dark: Colors::from_tokens(table(&["color", "dark"])?, &palette).map_err(Error)?,
            typography: Typography::from_tokens(table(&["typography"])?)
                .map_err(|e| Error(format!("typography: {e}")))?,
        })
    }
}

/// A partial tokens file to merge on top of the built-in tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Layer {
    pub label: String,
    pub source: String,
}

/// An error encountered while loading or compiling token data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

impl std::error::Error for Error {}

/// Merges the overlay table into the base table, recursively merging nested tables.
///
/// If a key exists in both tables, the value from the overlay table will overwrite
/// the value from the base table.
fn merge(base: &mut toml::Table, overlay: toml::Table) {
    for (key, value) in overlay {
        match (base.get_mut(&key), value) {
            (Some(toml::Value::Table(base_table)), toml::Value::Table(overlay_table)) => {
                merge(base_table, overlay_table)
            }
            (_, value) => {
                base.insert(key, value);
            }
        }
    }
}

/// Reads a layer from a file, returning None if the file is not found.
///
/// If the file is found but cannot be read, an error is added to the warnings list
/// and None is returned.
fn read_layer(path: &Path, warnings: &mut Vec<String>) -> Option<Layer> {
    match std::fs::read_to_string(path) {
        Ok(source) => Some(Layer {
            label: path.display().to_string(),
            source,
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            warnings.push(format!("{}: {error}; ignoring it", path.display()));
            None
        }
    }
}

/// Resolves a path from an XDG-style environment variable.
///
/// An absolute value from `var` takes precedence. If the variable is unset or it
/// contains a relative path, the path is resolved relative to `HOME` using `fallback`.
fn xdg_path(var: &str, fallback: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    std::env::var_os(var)
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| home.as_ref().map(|home| home.join(fallback)))
}
