use dirs::home_dir;
use path_absolutize::Absolutize;
use serde::Deserialize;
use serde::Serialize;
use std::path::Display;
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AbsolutePathBuf(PathBuf);

impl AbsolutePathBuf {
    fn maybe_expand_home_directory(path: &Path) -> PathBuf {
        let Some(path_str) = path.to_str() else {
            return path.to_path_buf();
        };
        if cfg!(not(target_os = "windows"))
            && let Some(home) = home_dir()
        {
            if path_str == "~" {
                return home;
            }
            if let Some(rest) = path_str.strip_prefix("~/") {
                let rest = rest.trim_start_matches('/');
                if rest.is_empty() {
                    return home;
                }
                return home.join(rest);
            }
        }
        path.to_path_buf()
    }

    pub fn resolve_path_against_base<P: AsRef<Path>, B: AsRef<Path>>(
        path: P,
        base_path: B,
    ) -> std::io::Result<Self> {
        let expanded = Self::maybe_expand_home_directory(path.as_ref());
        let absolute_path = expanded.absolutize_from(base_path.as_ref())?;
        Ok(Self(absolute_path.into_owned()))
    }

    pub fn from_absolute_path<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let expanded = Self::maybe_expand_home_directory(path.as_ref());
        let absolute_path = expanded.absolutize()?;
        Ok(Self(absolute_path.into_owned()))
    }

    pub fn current_dir() -> std::io::Result<Self> {
        Self::from_absolute_path(std::env::current_dir()?)
    }

    pub fn join<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Self> {
        Self::resolve_path_against_base(path, &self.0)
    }

    pub fn parent(&self) -> Option<Self> {
        self.0.parent().map(|parent| Self(parent.to_path_buf()))
    }

    pub fn as_path(&self) -> &Path {
        &self.0
    }

    pub fn into_path_buf(self) -> PathBuf {
        self.0
    }

    pub fn to_path_buf(&self) -> PathBuf {
        self.0.clone()
    }

    pub fn to_string_lossy(&self) -> std::borrow::Cow<'_, str> {
        self.0.to_string_lossy()
    }

    pub fn display(&self) -> Display<'_> {
        self.0.display()
    }
}

impl AsRef<Path> for AbsolutePathBuf {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl From<AbsolutePathBuf> for PathBuf {
    fn from(path: AbsolutePathBuf) -> Self {
        path.into_path_buf()
    }
}

impl TryFrom<&Path> for AbsolutePathBuf {
    type Error = std::io::Error;

    fn try_from(value: &Path) -> Result<Self, Self::Error> {
        Self::from_absolute_path(value)
    }
}

impl TryFrom<PathBuf> for AbsolutePathBuf {
    type Error = std::io::Error;

    fn try_from(value: PathBuf) -> Result<Self, Self::Error> {
        Self::from_absolute_path(value)
    }
}

impl TryFrom<&str> for AbsolutePathBuf {
    type Error = std::io::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::from_absolute_path(value)
    }
}

impl TryFrom<String> for AbsolutePathBuf {
    type Error = std::io::Error;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::from_absolute_path(value)
    }
}
