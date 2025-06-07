// src/tokenizer/progress.rs

// This module will be used when the "progressbar" feature is NOT enabled.
// It provides dummy implementations for ProgressBar and ProgressStyle.

use std::borrow::Cow;

pub struct ProgressBar;

impl ProgressBar {
    pub fn new(_length: u64) -> Self {
        Self {}
    }

    pub fn set_length(&self, _length: u64) {}
    pub fn set_message(&self, _message: impl Into<Cow<'static, str>>) {}
    pub fn finish(&self) {} // Corrected fnish to finish
    pub fn reset(&self) {}
    pub fn inc(&self, _inc: u64) {}
    pub fn set_style(&self, _style: ProgressStyle) {}
    pub fn set_position(&self, _pos: u64) {} // Added set_position
}

pub struct ProgressStyle;

impl ProgressStyle {
    pub fn default_bar() -> Self {
        Self {}
    }
    pub fn template(self, _template: &str) -> Result<Self, String> {
        Ok(self)
    }
}