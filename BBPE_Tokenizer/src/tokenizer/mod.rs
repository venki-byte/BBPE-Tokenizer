// src/tokenizer/mod.rs

// Declare all sub-modules that exist in the directory
pub mod added_vocabulary;
pub mod bpe; // Make sure bpe.rs is declared
pub mod bpe_trainer;
pub mod cache;
pub mod first_last_iterator;
pub mod parallelism;
pub mod pair;
pub mod pre_tokenizer;
pub mod progress; // Keep progress if it's for user-facing progress, not debugging
pub mod result; // We will adjust this file
pub mod word;

// Re-export important types/traits from their respective modules
// This makes them directly accessible via `tokenizer::SomeType` from outside `tokenizer` module
pub use added_vocabulary::AddedToken;
pub use bpe::BPE; // Re-export BPE model
pub use bpe::Rank;

// FIX 1: Correctly re-export Trainer.
// Trainer is defined in this 'mod.rs' (line 69), so we don't need to re-export it from bpe_trainer.
// We just need to make sure bpe_trainer's BpeTrainer and BpeTrainerBuilder are public.
pub use bpe_trainer::{BpeTrainer, BpeTrainerBuilder};
pub use pre_tokenizer::{ByteLevel, PreTokenizedString, PreTokenizer};
pub use result::{Error, Result}; // Your custom Error and Result aliases (adjusted in result.rs)
pub use pair::Pair; // Re-export the Pair type
pub use first_last_iterator::{FirstLastIterator, WithFirstLastIterator}; // Re-export the trait and its struct
pub use word::Word; // Re-export the Word struct

/// A trait defining the core behavior of a tokenizer Model.
/// Adjusted to match what `BPE` can currently easily provide.
// FIX 2: Add `get_vocab` and `get_vocab_r` to the Model trait.
// Also, if BPE::default() is meant to satisfy a Default bound, `Model` should require `Default`.
pub trait Model: Send + Sync + Default { // Added `+ Default`
    /// Tokenizes the given input string.
    fn tokenize(&self, text: &str, allowed_special: &std::collections::HashSet<&str>) -> Vec<crate::tokenizer::Rank>;

    /// Decodes a slice of token ranks back into bytes.
    fn decode_bytes(&self, tokens: &[crate::tokenizer::Rank]) -> std::result::Result<Vec<u8>, crate::tokenizer::bpe::DecodeKeyError>;

    /// Converts a token ID back to its string representation.
    fn id_to_token(&self, id: u32) -> Option<String>;

    /// Converts a token string to its ID.
    fn token_to_id(&self, token: &str) -> Option<u32>;

    /// Returns the vocabulary size of the model.
    fn get_vocab_size(&self) -> usize;

    // ADDED: Methods for getting vocab and reverse vocab.
    // Assuming these should return references to HashMap<String, u32> and HashMap<u32, String> respectively.
    fn get_vocab(&self) -> &rustc_hash::FxHashMap<String, u32>; // <-- Change here
    fn get_vocab_r(&self) -> &rustc_hash::FxHashMap<u32, String>; // <-- Change here
}

/// Represents a single token with its ID, content, and original offsets.
// Removed `Debug` from `derive` attributes if it's only for debugging and not used otherwise.
// If you need `Debug` for other reasons (e.g., in tests or for logging non-performance-critical paths), keep it.
// For performance-critical paths, avoid Debug printing.
#[derive(Clone, PartialEq, Eq, Hash)] // Removed Debug
pub struct Token {
    pub id: u32,
    pub value: String,
    pub offsets: (usize, usize), // (start_byte, end_byte)
}

impl Token {
    pub fn new(id: u32, value: String, offsets: (usize, usize)) -> Self {
        Token { id, value, offsets }
    }
}


/// A trait defining the behavior of a tokenizer Trainer.
pub trait Trainer: Send + Sync {
    /// The type of Model this trainer can produce.
    type Model: Model;

    /// Trains the tokenizer model.
    /// Returns a list of special tokens that were added during training.
    fn train(&self, model: &mut Self::Model) -> Result<Vec<AddedToken>>;

    /// Feeds data to the trainer for training.
    /// `iterator`: An iterator over the training sequences (e.g., sentences).
    /// `process`: A function that preprocesses each sequence into a list of words or strings
    ///            that the trainer can understand (e.g., by splitting on whitespace).
    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync;

    /// Indicates whether training progress should be displayed.
    // Kept this as it seems like a user-facing feature rather than a debug-only one.
    fn should_show_progress(&self) -> bool;

    // Any other common trainer methods can go here.
}