// Pretokenization - Standalone Version (no external file dependencies)

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use regex::Regex;

// Converts bytes to visible Unicode characters for display/debug/tokenization
// This function remains largely as is, as it's part of the core logic for byte-level tokenization.
// The `panic!` in `unwrap_or_else` is appropriate here, as an invalid Unicode scalar value
// indicates a fundamental problem that cannot be recovered from gracefully during mapping.
pub(crate) fn bytes_char() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = vec![];
    bs.extend(b'!'..=b'~');
    bs.extend(b'\xA1'..=b'\xAC');
    bs.extend(b'\xAE'..=b'\xFF');

    let mut cs: Vec<u32> = bs.iter().map(|i| *i as u32).collect();

    let mut n = 0;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    bs.into_iter()
        .zip(cs)
        .map(|(f, t)| {
            let ch = std::char::from_u32(t)
                .unwrap_or_else(|| panic!("Invalid Unicode scalar value: {}", t));
            (f, ch)
        })
        .collect()
}


// The regex pattern for splitting. `unwrap()` here is fine as the pattern is static
// and a failure to compile it indicates a fundamental, unrecoverable programming error.
pub static RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}| ?[^\s\p{L}\p{N}]+|\s+").unwrap()
});

static BYTES_CHAR: LazyLock<HashMap<u8, char>> = LazyLock::new(bytes_char);

// Removed `Debug` derive from ByteLevel as it's a zero-sized struct and `Debug` isn't critical for it.
// If you need `Debug` for broader testing, you can add it back.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct ByteLevel;

// ADDED `#[derive(Debug)]` HERE
#[derive(Debug)]
pub enum SplitDelimiterBehavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

pub struct NormalizedString {
    content: String,
}

impl NormalizedString {
    pub fn get(&self) -> &str {
        &self.content
    }

    // This method has a potential optimization: `to_string()` creates new allocations.
    // For extreme performance, one might consider passing slices or using rope-like structures,
    // but for typical tokenizer pre-processing, `to_string()` is usually acceptable.
    // The `Err` for unimplemented behaviors is crucial for correctness.
    pub fn split(&mut self, pattern: &Regex, behavior: SplitDelimiterBehavior) -> Result<Vec<NormalizedString>, String> {
        let mut result = Vec::new(); // Use `Vec::new()` directly
        let text = self.get();
        let mut last_end = 0;
        for mat in pattern.find_iter(text) {
            let (start, end) = (mat.start(), mat.end());
            match behavior {
                SplitDelimiterBehavior::Contiguous => {
                    if last_end < start {
                        result.push(NormalizedString { content: text[last_end..start].to_string() });
                    }
                    result.push(NormalizedString { content: text[start..end].to_string() });
                    last_end = end;
                }
                // Panicking here is appropriate if other behaviors are truly unsupported
                // and represent an invalid state for this particular implementation.
                _ => panic!("Unsupported SplitDelimiterBehavior: {:?}", behavior),
            }
        }
        if last_end < text.len() {
            result.push(NormalizedString { content: text[last_end..].to_string() });
        }
        Ok(result)
    }

    // `_offset` parameter is unused, consider removing if it's not needed by future transformations.
    // If it's a placeholder, keep it.
    pub fn transform(&mut self, transformations: Vec<(char, isize)>, _offset: usize) {
        // This collect operation creates a new string, which is standard for string manipulation.
        self.content = transformations.into_iter().map(|(c, _)| c).collect();
    }
}

pub struct PreTokenizedString {
    splits: Vec<NormalizedString>,
}

impl PreTokenizedString {
    pub fn new(text: &str) -> Self {
        Self {
            splits: vec![NormalizedString { content: text.to_string() }],
        }
    }

    // This method is good for consuming the processed splits.
    pub fn take_splits(self) -> Vec<NormalizedString> {
        self.splits
    }

    // These getters are fine.
    pub fn get_mut_splits(&mut self) -> &mut Vec<NormalizedString> {
        &mut self.splits
    }

    pub fn get_splits(&self) -> &Vec<NormalizedString> {
        &self.splits
    }

    // Optimized `split` to avoid intermediate `Vec` allocations by pre-allocating.
    // Using `drain` is efficient for moving items out of the old vector.
    pub fn split<F>(&mut self, mut split_fn: F) -> Result<(), String>
    where
        F: FnMut(usize, NormalizedString) -> Result<Vec<NormalizedString>, String>,
    {
        let old_splits = std::mem::take(&mut self.splits); // Take ownership to clear `self.splits`
        let mut new_splits_capacity = 0;
        // Estimate capacity for new_splits to reduce reallocations.
        // This is a heuristic; exact capacity is hard to predict.
        // Assume each split creates at least one new split, maybe more.
        if !old_splits.is_empty() {
             new_splits_capacity = old_splits.len() * 2; // A common heuristic
        }
        let mut new_splits = Vec::with_capacity(new_splits_capacity);

        for (i, split) in old_splits.into_iter().enumerate() {
            new_splits.extend(split_fn(i, split)?);
        }
        self.splits = new_splits;
        Ok(())
    }

    pub fn normalize<F>(&mut self, mut normalize_fn: F) -> Result<(), String>
    where
        F: FnMut(&mut NormalizedString) -> Result<(), String>,
    {
        for split in &mut self.splits {
            normalize_fn(split)?;
        }
        Ok(())
    }
}

pub trait PreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<(), String>;
}

impl ByteLevel {
    pub fn new() -> Self {
        ByteLevel
    }

    pub fn alphabet() -> HashSet<char> {
        BYTES_CHAR.values().copied().collect()
    }
}

impl PreTokenizer for ByteLevel {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<(), String> {
        let re_ref = &*RE;
        pretokenized.split(|_, mut normalized| {
            normalized.split(re_ref, SplitDelimiterBehavior::Contiguous)
        })?;

        pretokenized.normalize(|normalized| {
            let s = normalized.get();
            // Estimate capacity for `transformations` upfront to prevent reallocations.
            // Each char can map to multiple bytes, so a conservative estimate is `s.len() * 4` (max utf8 bytes)
            // or more accurately, `s.as_bytes().len()`.
            let mut transformations = Vec::with_capacity(s.as_bytes().len());
            let mut i = 0;
            for ch in s.chars() {
                let len = ch.len_utf8();
                let bytes = &s.as_bytes()[i..i + len];
                i += len;
                for (_j, b) in bytes.iter().enumerate() { // `_j` indicates it's unused
                    // Removed `shift` as it's unused in `normalized.transform`
                    let mapped = *BYTES_CHAR.get(b)
                        // Using `expect` here is appropriate if a missing byte mapping
                        // indicates a truly unrecoverable and unexpected internal error.
                        .expect("Missing byte mapping for a byte during pre-tokenization. This indicates a severe internal error.");
                    transformations.push((mapped, 0)); // `0` for `shift` since it's unused
                }
            }
            normalized.transform(transformations, 0); // `0` for `offset` since it's unused
            Ok(())
        })
    }
}