// src/tokenizer/bpe.rs

use std::collections::HashSet;
use std::num::NonZeroU64;
use std::thread; // Make sure 'thread' is imported
use fancy_regex::Regex;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use bstr; // bstr seems unused in the provided snippets for now, but kept if needed elsewhere

pub type Rank = u32;

use crate::tokenizer::Model;
use crate::tokenizer::Error; // Import your custom Error type

#[cfg(feature = "python")]
mod py;

#[cfg(not(feature = "progressbar"))] // Assuming progressbar is in progress.rs, not bpe.rs
mod progressbar {
    use std::borrow::Cow;
    pub struct ProgressBar;
    impl ProgressBar {
        pub fn new(_length: u64) -> Self {
            Self {}
        }

        pub fn set_length(&self, _length: u64) {}
        pub fn set_message(&self, _message: impl Into<Cow<'static, str>>) {}
        pub fn finish(&self) {}
        pub fn reset(&self) {}
        pub fn inc(&self, _inc: u64) {}
        pub fn set_style(&self, _style: ProgressStyle) {}
    }

    pub struct ProgressStyle {}
    impl ProgressStyle {
        pub fn default_bar() -> Self {
            Self {}
        }
        pub fn template(self, _template: &str) -> Result<Self, String> {
            Ok(self)
        }
    }
}
#[cfg(not(feature = "progressbar"))]
pub(crate) use progressbar::{ProgressBar, ProgressStyle};


// --- Error Types ---
#[derive(Debug, Clone)]
pub struct DecodeKeyError {
    pub token: Rank,
}

impl std::fmt::Display for DecodeKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Invalid token for decoding: {}", self.token)
    }
}

impl std::error::Error for DecodeKeyError {}

#[derive(Debug, Clone)]
pub struct DecodeError {
    pub message: String,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Could not decode tokens: {}", self.message)
    }
}

impl std::error::Error for DecodeError {}


// --- Thread-Local Regex Hashing ---
const MAX_NUM_THREADS: usize = 128; // Using a const

// This is a common workaround for `std::thread::ThreadId` not implementing `Hash`
// or `Copy` in a stable way that allows it to be used as a HashMap key directly
// or for simple numeric hashing.
struct FakeThreadId(NonZeroU64);

fn hash_current_thread() -> usize {
    // This transmutes `std::thread::ThreadId` to `FakeThreadId` (which wraps `NonZeroU64`)
    // to get a numeric representation for hashing. This is an unsafe operation,
    // but a common pattern for this specific use case (thread-local storage).
    // The `_` constant asserts are for compile-time size checks, ensuring the transmute is safe.
    const _: [u8; 8] = [0; std::mem::size_of::<std::thread::ThreadId>()];
    const _: [u8; 8] = [0; std::mem::size_of::<FakeThreadId>()];
    let x = unsafe {
        std::mem::transmute::<std::thread::ThreadId, FakeThreadId>(thread::current().id()).0
    };
    u64::from(x) as usize
}

// --- BPE Struct Definition ---
#[cfg_attr(feature = "python", pyclass)]
#[derive(Debug, Clone)]
pub struct BPE {
    pub encoder: HashMap<Vec<u8>, Rank>,
    pub special_tokens_encoder: HashMap<String, Rank>,
    decoder: HashMap<Rank, Vec<u8>>,
    special_tokens_decoder: HashMap<Rank, Vec<u8>>,
    regex_tls: Vec<Regex>,
    special_regex_tls: Vec<Regex>,
    sorted_token_bytes: Vec<Vec<u8>>,
    pub vocab_map: HashMap<String, u32>,
    pub vocab_r_map: HashMap<u32, String>,
    pub merges: Vec<(String, String)>,
}

// --- BPE Implementation for Model Trait ---
impl Model for BPE {
    fn tokenize(&self, text: &str, allowed_special: &HashSet<&str>) -> Vec<Rank> {
        self.encode(text, allowed_special).0
    }

    fn decode_bytes(&self, tokens: &[Rank]) -> Result<Vec<u8>, crate::tokenizer::bpe::DecodeKeyError> {
        // This calls the actual `decode_bytes` method of the BPE struct
        self.decode_bytes(tokens)
    }

    fn get_vocab_size(&self) -> usize {
        self.encoder.len() + self.special_tokens_encoder.len()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        if let Some(bytes) = self.decoder.get(&id) {
            String::from_utf8(bytes.clone()).ok()
        } else if let Some(bytes) = self.special_tokens_decoder.get(&id) {
            String::from_utf8(bytes.clone()).ok()
        } else {
            None
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        if let Some(id) = self.encoder.get(token.as_bytes()) {
            Some(*id)
        } else {
            self.special_tokens_encoder.get(token).copied()
        }
    }

    // These methods are also defined directly on BPE, the trait requires them to be callable from the trait object
    fn get_vocab(&self) -> &HashMap<String, u32> {
        &self.vocab_map
    }

    fn get_vocab_r(&self) -> &HashMap<u32, String> {
        &self.vocab_r_map
    }
}

// --- Internal Byte Pair Merge Logic ---
fn _byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    let mut parts = Vec::with_capacity(piece.len() + 1);

    // Initialize `parts` with indices and their respective ranks (or MAX if no rank exists)
    // The logic here finds the smallest rank (highest priority merge)
    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX)); // Add last character's index with max rank
    parts.push((piece.len(), Rank::MAX));     // Sentinel for end of piece

    // Helper closure to get rank for a potential new merge after a merge has occurred
    let get_rank = {
        #[inline(always)] // Optimize for inlining
        |parts: &Vec<(usize, Rank)>, i: usize| {
            if (i + 3) < parts.len() {
                // If there are enough parts to form a triplet after a merge, get its rank
                *ranks
                    .get(&piece[parts[i].0..parts[i + 3].0])
                    .unwrap_or(&Rank::MAX)
            } else {
                Rank::MAX
            }
        }
    };

    // Main merge loop
    while min_rank.0 != Rank::MAX {
        let i = min_rank.1; // Index of the highest priority merge

        // Update ranks of surrounding pairs, as merging affects them
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1); // Remove the second part of the merged pair

        // Find the next highest priority merge
        min_rank = (Rank::MAX, usize::MAX);
        for (idx, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, idx);
            }
        }
    }
    parts
}

pub fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    if piece.len() == 1 {
        // Single byte pieces are directly mapped
        return vec![ranks[piece]];
    }
    // Perform merges and then map resulting byte slices to their ranks
    _byte_pair_merge(ranks, piece)
        .windows(2) // Create windows of two parts to get (start_index, end_index)
        .map(|part| ranks[&piece[part[0].0..part[1].0]]) // Map the byte slice to its rank
        .collect()
}

pub fn byte_pair_split<'a>(piece: &'a [u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<&'a [u8]> {
    assert!(piece.len() > 1);
    // Perform merges and then collect the resulting byte slices
    _byte_pair_merge(ranks, piece)
        .windows(2)
        .map(|part| &piece[part[0].0..part[1].0])
        .collect()
}

// --- BPE Main Implementation ---
impl BPE {
    // Helper to get thread-local regex instance
    fn _get_tl_regex(&self) -> &Regex {
        &self.regex_tls[hash_current_thread() % MAX_NUM_THREADS]
    }

    // Helper to get thread-local special regex instance
    fn _get_tl_special_regex(&self) -> &Regex {
        &self.special_regex_tls[hash_current_thread() % MAX_NUM_THREADS]
    }

    // Decodes a slice of token ranks back into a sequence of bytes
    pub fn decode_bytes(&self, tokens: &[Rank]) -> Result<Vec<u8>, DecodeKeyError> {
        let mut ret = Vec::with_capacity(tokens.len() * 2); // Pre-allocate for efficiency
        for &token in tokens {
            let token_bytes = match self.decoder.get(&token) {
                Some(bytes) => bytes,
                // If not in regular decoder, check special tokens decoder
                None => self
                    .special_tokens_decoder
                    .get(&token)
                    .ok_or(DecodeKeyError { token })?, // Return error if token not found
            };
            ret.extend_from_slice(token_bytes); // Use extend_from_slice for Vec<u8>
        }
        Ok(ret)
    }

    // Encodes text without considering special tokens
    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        let regex = self._get_tl_regex();
        let mut ret = vec![];
        for mat in regex.find_iter(text) {
            let piece = mat.unwrap().as_str().as_bytes(); // Get the byte slice of the match
            match self.encoder.get(piece) {
                Some(token) => ret.push(*token), // If piece is a known token, use its rank
                None => ret.extend(byte_pair_encode(piece, &self.encoder)), // Otherwise, apply BPE
            }
        }
        ret
    }

    // Main encoding function, handles both ordinary and special tokens
    pub fn encode(&self, text: &str, allowed_special: &HashSet<&str>) -> (Vec<Rank>, usize) {
        let special_regex = self._get_tl_special_regex();
        let regex = self._get_tl_regex();
        let mut ret = vec![];

        let mut start = 0;
        let mut last_piece_token_len = 0; // Tracks length of tokens from the last non-special piece

        loop {
            let mut next_special_match_option;
            let mut search_start_pos = start;

            // Find the next special token that is allowed
            loop {
                next_special_match_option = special_regex.find_from_pos(text, search_start_pos).unwrap();
                match next_special_match_option {
                    Some(m) => {
                        // If the matched special token is allowed, break and process it
                        if allowed_special.contains(&text[m.start()..m.end()]) {
                            break;
                        }
                        // Otherwise, skip this special token and search from the character after it
                        search_start_pos = m.start() + 1;
                    }
                    None => break, // No more special tokens found
                }
            }
            
            // Define the end of the current ordinary text segment (before the next special token)
            let end_ordinary_segment = next_special_match_option.map_or(text.len(), |m| m.start());

            // Process the ordinary text segment
            for mat in regex.find_iter(&text[start..end_ordinary_segment]) {
                let piece = mat.unwrap().as_str().as_bytes();
                if let Some(token) = self.encoder.get(piece) {
                    last_piece_token_len = 1;
                    ret.push(*token);
                    continue;
                }
                let tokens = byte_pair_encode(piece, &self.encoder);
                last_piece_token_len = tokens.len();
                ret.extend(tokens); // Use extend for Vec<Rank>
            }

            // Process the special token if one was found
            match next_special_match_option {
                Some(m) => {
                    let piece = m.as_str();
                    // Special tokens must exist in special_tokens_encoder
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    start = m.end(); // Move start to after the special token
                    last_piece_token_len = 0; // Reset as special token breaks piece
                }
                None => break, // No more text to process
            }
        }

        (ret, last_piece_token_len)
    }

    // Helper to adjust `last_piece_token_len` for trailing whitespace
    fn _increase_last_piece_token_len(
        &self,
        mut tokens: Vec<Rank>, // Takes ownership to modify
        mut last_piece_token_len: usize,
    ) -> (Vec<Rank>, usize) {
        {
            let token_is_all_space = |token| {
                self.decoder
                    .get(token)
                    .map(|token_bytes| {
                        token_bytes
                            .iter()
                            .rev() // Iterate in reverse to check trailing spaces first
                            .all(|&b| [b' ', b'\n', b'\t'].contains(&b)) // Check if all bytes are whitespace
                    })
                    .unwrap_or(false)
            };
            if last_piece_token_len > 0
                && token_is_all_space(&tokens[tokens.len() - last_piece_token_len])
            {
                // If the last piece is all space, extend `last_piece_token_len` backward
                // to include previous tokens that are also all space.
                while (last_piece_token_len < tokens.len())
                    && token_is_all_space(&tokens[tokens.len() - last_piece_token_len - 1])
                {
                    last_piece_token_len += 1;
                }
            }
        }
        debug_assert!(last_piece_token_len <= tokens.len());

        (tokens, last_piece_token_len)
    }

    // Encodes text and returns unstable completions (for language model generation)
    pub fn _encode_unstable_native(
        &self,
        text: &str,
        allowed_special: &HashSet<&str>,
    ) -> (Vec<Rank>, HashSet<Vec<Rank>>) {
        let (mut tokens, last_piece_token_len) = self.encode(text, allowed_special);
        if last_piece_token_len == 0 {
            return (tokens, HashSet::new()); // No unstable piece
        }
        let (final_tokens, adjusted_last_piece_token_len) =
            self._increase_last_piece_token_len(tokens, last_piece_token_len);
        tokens = final_tokens; // Update tokens with the possibly re-adjusted last piece

        // Decode the unstable bytes
        let unstable_bytes = self
            .decode_bytes(&tokens[tokens.len() - adjusted_last_piece_token_len..])
            .unwrap(); // Using unwrap here assumes decode_bytes won't fail for valid tokens
        tokens.truncate(tokens.len() - adjusted_last_piece_token_len); // Remove unstable piece from tokens

        let mut completions = HashSet::new();
        if unstable_bytes.is_empty() {
            return (tokens, completions);
        }

        // Find completions by matching prefixes of `unstable_bytes` with sorted tokens
        let mut point = self
            .sorted_token_bytes
            .partition_point(|x| x.as_slice() < unstable_bytes.as_slice());
        while point < self.sorted_token_bytes.len()
            && self.sorted_token_bytes[point].starts_with(&unstable_bytes)
        {
            completions.insert(vec![
                self.encoder[self.sorted_token_bytes[point].as_slice()],
            ]);
            point += 1;
        }

        // Find completions by splitting `unstable_bytes` and matching suffix
        for i in 1..unstable_bytes.len() {
            let prefix = &unstable_bytes[..i];
            let suffix = &unstable_bytes[i..];
            let mut point = self
                .sorted_token_bytes
                .partition_point(|x| x.as_slice() < suffix);
            while point < self.sorted_token_bytes.len()
                && self.sorted_token_bytes[point].starts_with(suffix)
            {
                let possibility = [prefix, self.sorted_token_bytes[point].as_slice()].concat();
                let encoded = match std::str::from_utf8(&possibility) {
                    Ok(s) => self.encode_ordinary(s),
                    Err(_) => byte_pair_encode(&possibility, &self.encoder),
                };
                let mut seq = Vec::new();
                let mut seq_len = 0;
                for token in encoded {
                    seq.push(token);
                    seq_len += self.decoder[&token].len();
                    if seq_len >= unstable_bytes.len() {
                        break;
                    }
                }
                completions.insert(seq);
                point += 1;
            }
        }

        // Handle specific case for trailing whitespace
        if unstable_bytes.len() > 1 {
            let last_decoded = bstr::decode_last_utf8(unstable_bytes.as_slice());
            if unstable_bytes.len() - last_decoded.1 > 0
                && last_decoded.0.map_or(false, |c| c.is_whitespace())
            {
                let mut reencoded = byte_pair_encode(
                    &unstable_bytes[..unstable_bytes.len() - last_decoded.1],
                    &self.encoder,
                );
                reencoded.extend(byte_pair_encode(
                    &unstable_bytes[unstable_bytes.len() - last_decoded.1..],
                    &self.encoder,
                ));
                completions.insert(reencoded);
            }
        }

        (tokens, completions)
    }

    // Constructor for BPE, taking iterators for flexibility
    pub fn new<E, SE>(
        encoder: E,
        special_tokens_encoder: SE,
        pattern: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> // Return type using Box<dyn Error>
    where
        E: IntoIterator<Item = (Vec<u8>, Rank)>,
        SE: IntoIterator<Item = (String, Rank)>,
    {
        Self::new_internal(
            HashMap::from_iter(encoder),
            HashMap::from_iter(special_tokens_encoder),
            pattern,
        )
    }

    // Internal constructor that works with actual HashMaps
    fn new_internal(
        encoder: HashMap<Vec<u8>, Rank>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> { // Return type using Box<dyn Error>
        let regex = Regex::new(pattern)?; // Regex compilation can fail

        let special_regex = {
            let parts = special_tokens_encoder
                .keys()
                .map(|s| fancy_regex::escape(s)) // Escape special characters for regex
                .collect::<Vec<_>>();
            // Join parts with '|' to form a regex that matches any of the special tokens
            Regex::new(&parts.join("|"))?
        };

        // Create reverse mapping (decoder) from encoder
        let decoder: HashMap<Rank, Vec<u8>> =
            encoder.iter().map(|(k, v)| (*v, k.clone())).collect();

        // Ensure encoder and decoder have same number of elements (no duplicate ranks)
        assert!(
            encoder.len() == decoder.len(),
            "Encoder and decoder must be of equal length; maybe you had duplicate token indices in your encoder?"
        );

        // Create reverse mapping for special tokens
        let special_tokens_decoder: HashMap<Rank, Vec<u8>> = special_tokens_encoder
            .iter()
            .map(|(k, v)| (*v, k.as_bytes().to_vec()))
            .collect();

        // Populate vocab_map (string to ID) and vocab_r_map (ID to string)
        let mut vocab_map = HashMap::default();
        let mut vocab_r_map = HashMap::default();

        for (bytes, &id) in encoder.iter() {
            let s = String::from_utf8(bytes.clone())
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?; // Handle UTF-8 conversion error
            vocab_map.insert(s.clone(), id);
            vocab_r_map.insert(id, s);
        }

        for (s, &id) in special_tokens_encoder.iter() {
            vocab_map.insert(s.clone(), id);
            vocab_r_map.insert(id, s.clone());
        }

        // Create and sort a list of token bytes for efficient prefix/suffix matching
        let mut sorted_token_bytes: Vec<Vec<u8>> = encoder.keys().cloned().collect();
        sorted_token_bytes.sort();

        Ok(Self {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            regex_tls: (0..MAX_NUM_THREADS).map(|_| regex.clone()).collect(), // Clone regex for each thread slot
            special_regex_tls: (0..MAX_NUM_THREADS)
                .map(|_| special_regex.clone())
                .collect(), // Clone special regex
            sorted_token_bytes,
            vocab_map,
            vocab_r_map,
            merges: Vec::new(), // Merges are populated during training
        })
    }

    // Returns a HashSet of string slices of special tokens
    pub fn special_tokens(&self) -> HashSet<&str> {
        self.special_tokens_encoder
            .keys()
            .map(|s| s.as_str())
            .collect()
    }

    // Encodes text including special tokens that are implicitly allowed
    pub fn encode_with_special_tokens(&self, text: &str) -> Vec<Rank> {
        let allowed_special = self.special_tokens(); // Get all special tokens
        self.encode(text, &allowed_special).0
    }

    // Returns the total vocabulary size (regular + special tokens)
    pub fn get_vocab_size(&self) -> usize {
        self.encoder.len() + self.special_tokens_encoder.len()
    }

    // Public getters for vocab and merges
    pub fn get_vocab(&self) -> &HashMap<String, u32> {
        &self.vocab_map
    }

    pub fn get_merges(&self) -> &Vec<(String, String)> {
        &self.merges
    }
}

// --- Default Implementation for BPE ---
impl Default for BPE {
    fn default() -> Self {
        BPE::new(
            HashMap::default(), // Empty encoder
            HashMap::default(), // Empty special tokens encoder
            // Default GPT-2 like tokenizer pattern
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        ).unwrap_or_else(|e| panic!("Failed to create default BPE: {}", e))
    }
}