// src/tokenizer/result.rs

use std::error::Error as StdError;
use std::fmt::{self, Display, Formatter};

#[derive(Debug)]
pub enum Error {
    GenericError(String),
    IoError(std::io::Error),
    JsonError(serde_json::Error),
    // Add other specific error types as needed
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Error::GenericError(msg) => write!(f, "Generic error: {}", msg),
            Error::IoError(err) => write!(f, "IO error: {}", err),
            Error::JsonError(err) => write!(f, "JSON error: {}", err),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::GenericError(_) => None,
            Error::IoError(err) => Some(err),
            Error::JsonError(err) => Some(err),
        }
    }
}

impl From<String> for Error {
    fn from(err: String) -> Self {
        Error::GenericError(err)
    }
}

impl From<&str> for Error {
    fn from(err: &str) -> Self {
        Error::GenericError(err.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IoError(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::JsonError(err)
    }
}

// Keep this if you specifically need to handle Send + Sync errors
impl From<Box<dyn StdError + Send + Sync>> for Error {
    fn from(err: Box<dyn StdError + Send + Sync>) -> Self {
        Error::GenericError(format!("Caught boxed error: {}", err))
    }
}

// ADD THIS NEW `From` IMPLEMENTATION
// This handles cases where the error is just Box<dyn StdError>
impl From<Box<dyn StdError>> for Error {
    fn from(err: Box<dyn StdError>) -> Self {
        Error::GenericError(format!("Caught boxed error (no Send/Sync): {}", err))
    }
}


pub type Result<T> = std::result::Result<T, Error>;