/*

This code provides a way to extend a tokenizer's vocabulary after a model has been trained, which is crucial for fine-tuning or adapting to new domains without retraining the entire model.
The AddedToken struct defines properties for custom tokens, including their content, whether they should match whole words (single_word), handle surrounding whitespace (lstrip, rstrip), be normalized, and if they are considered "special."
The Default implementation for AddedToken sets up a basic token. Notably, special is true and normalized is false by default, indicating it's initially designed for special, literal-matching tokens.
The AddedToken::from constructor and builder methods (single_word, lstrip, etc.) allow for flexible configuration of AddedToken properties, overriding the defaults to suit specific token behaviors.
The AddedVocabulary struct manages these added tokens, maintaining mappings between token content and their IDs, and also storing reverse mappings to retrieve token properties from IDs.
It distinguishes between "special" tokens (like <s>, </s>) and "classic" added tokens, handling them separately for specific processing needs, such as during decoding.
The core mechanism for finding added tokens in text relies on two Aho-Corasick automata (tries): one for non-normalized matches and another for normalized matches. This ensures efficient pattern searching.
When tokens are added via add_tokens, the internal tries are refreshed to incorporate the new patterns, ensuring the tokenizer can correctly identify them in subsequent text.
The find_matches function is central to token extraction; it iterates through text, finds matches for added tokens, and applies the single_word, lstrip, and rstrip rules to determine the final boundaries of the token.
The extract_and_normalize method first searches for non-normalized added tokens, then normalizes the remaining parts of the text and searches for normalized added tokens, creating a PreTokenizedString with the identified added tokens and other text segments.


*/


use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use regex::Regex;
use serde::{ser::SerializeSeq, Deserialize, Serialize, Serializer};
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

// Only bring in `Model` and `Token` from `super` as they are still relevant.
// Removed: normalizer::Range, NormalizedString, Normalizer, Offsets, PreTokenizedString
use super::{Model, Token};


/// Represent a token added by the user on top of the existing Model vocabulary.
/// AddedToken can be configured to specify the behavior they should have in various situations
/// like:
///   - Whether they should only match single words
///   - Whether to include any whitespace on its left or right
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AddedToken {
    /// The content of the added token
    pub content: String,
    /// Whether this token must be a single word or can break words
    pub single_word: bool,
    /// Whether this token should strip whitespaces on its left
    pub lstrip: bool,
    /// Whether this token should strip whitespaces on its right
    pub rstrip: bool,
    /// Whether this token should be normalized. This will now always be false for special tokens.
    pub normalized: bool,
    /// Whether this token is special
    pub special: bool,
}

impl AddedToken {
    /// Build this token from the given content, specifying if it is intended to be a
    /// special token. Special tokens are not normalized by default.
    pub fn from<S: Into<String>>(content: S, special: bool) -> Self {
        Self {
            content: content.into(),
            // Ensure normalized is false for special tokens in this context
            normalized: false, // Always false now if we are only handling non-normalized special tokens
            special,
            ..Default::default()
        }
    }
    /// Specify whether this token should only match on whole single words, and never
    /// part of a word.
    #[must_use]
    pub fn single_word(mut self, single_word: bool) -> Self {
        self.single_word = single_word;
        self
    }
    /// Specify whether this token should include all the whitespaces on its left, in
    /// order to strip them out.
    #[must_use]
    pub fn lstrip(mut self, lstrip: bool) -> Self {
        self.lstrip = lstrip;
        self
    }
    /// Specify whether this token should include all the whitespaces on its right, in
    /// order to strip them out.
    #[must_use]
    pub fn rstrip(mut self, rstrip: bool) -> Self {
        self.rstrip = rstrip;
        self
    }
    /// Specify whether this token should be normalized and match against its normalized
    /// version in the input text. This will now always be false for special tokens.
    #[must_use]
    pub fn normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }
    /// Specify whether this token is special, meaning if it should be skipped when decoding
    #[must_use]
    pub fn special(mut self, special: bool) -> Self {
        self.special = special;
        self
    }
}
impl Default for AddedToken {
    fn default() -> Self {
        Self {
            content: String::new(),
            single_word: true,
            lstrip: false,
            rstrip: false,
            normalized: false, // Changed to false by default for this specialized use case
            special: true,
        }
    }
}
// AddedTokens can be updated if value changed
impl std::hash::Hash for AddedToken {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.content.hash(state);
    }
}

type MatchingSet = (AhoCorasick, Vec<u32>);

// These regexes are still useful for `single_word`, `lstrip`, `rstrip` logic.
static STARTS_WITH_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\w").unwrap());
static ENDS_WITH_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w$").unwrap());
static RIGHTMOST_SPACE_AT_START: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*").unwrap());
static LEFTMOST_SPACE_AT_END: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s*$").unwrap());

fn ends_with_word(sentence: &str) -> bool {
    ENDS_WITH_WORD.is_match(sentence)
}

fn starts_with_word(sentence: &str) -> bool {
    STARTS_WITH_WORD.is_match(sentence)
}

fn space_leftmost_at_end(sentence: &str) -> usize {
    if let Some(match_) = LEFTMOST_SPACE_AT_END.find(sentence) {
        match_.start()
    } else {
        sentence.len()
    }
}
fn space_rightmost_at_start(sentence: &str) -> usize {
    if let Some(match_) = RIGHTMOST_SPACE_AT_START.find(sentence) {
        match_.end()
    } else {
        0
    }
}
///
/// A vocabulary built on top of the Model to handle **non-normalized special tokens**.
///
/// This provides a way to add new vocabulary to a Tokenizer that has already been trained,
/// in a previous process, maybe by someone else. This is especially interesting in the case
/// of fine-tunings, where we want to finetune a model while adding some new functionalities
/// using some new special tokens, or maybe add some tokens in the case of unknown tokens, etc.
///
/// One of the reasons we need to handle these tokens outside of the model is simply that
/// for many models, it is not possible to add new tokens after the training process. For example,
/// using BPE, the training process generates merges pairs along the vocabulary, and any token
/// in the vocabulary can be decomposed in other tokens, down to the original alphabet. If we
/// were to add new tokens after this training process, we couldn't make sure the merges pairs
/// exist as required.
///
#[derive(Clone, Debug)]
pub struct AddedVocabulary {
    /// Contains the mapping from String (token content) to ID. This map contains both special
    /// tokens and classic added tokens that were added to the this vocabulary.
    added_tokens_map: HashMap<String, u32>,
    /// Contains the mapping from ID to AddedToken for all the added tokens, both special
    /// and classic.
    added_tokens_map_r: HashMap<u32, AddedToken>,

    /// Contains only the classic AddedToken, in the specific order the user gave them.
    added_tokens: Vec<AddedToken>,
    /// Contains only the special AddedToken, in the specific order the user gave them.
    special_tokens: Vec<AddedToken>,

    /// A Set, containing all the special token for easy access while decoding. This let's
    /// us remove them easily with an O(1) complexity.
    special_tokens_set: HashSet<String>,

    /// An AhoCorasick automaton used to efficiently find non-normalized patterns (our special tokens).
    split_trie: MatchingSet,
    // Removed: split_normalized_trie
    // Removed: NormalizedString, Normalizer, Offsets, PreTokenizedString related imports and fields

    /// Whether or not special tokens should be splitted when encoding. This is equivalent to ignoring them
    encode_special_tokens: bool,
}

impl AddedVocabulary {
    pub fn new() -> Self {
        let trie = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build::<_, &&[u8]>([])
            .expect("The trie should build correctly");
        // Removed: normalized_trie initialization
        Self {
            added_tokens_map: HashMap::new(),
            added_tokens_map_r: HashMap::new(),
            added_tokens: vec![],
            special_tokens: vec![],
            special_tokens_set: HashSet::new(),
            split_trie: (trie, vec![]),
            // Removed: split_normalized_trie field
            encode_special_tokens: false,
        }
    }
    /// Size of the additional vocabulary
    #[allow(dead_code)] // Suppress the "method is never used" warning
    pub fn len(&self) -> usize {
        self.added_tokens_map.len()
    }

    /// Whether or not this vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.added_tokens_map.is_empty()
    }

    /// Get the additional vocabulary
    pub fn get_vocab(&self) -> &HashMap<String, u32> {
        &self.added_tokens_map
    }

    /// Get the additional vocabulary with the AddedTokens
    pub fn get_added_tokens_decoder(&self) -> &HashMap<u32, AddedToken> {
        &self.added_tokens_map_r
    }

    /// Get the id matching one of our token if it exists
    pub fn token_to_id(&self, token: &str, model: &impl Model) -> Option<u32> {
        self.added_tokens_map
            .get(token)
            .copied()
            .or_else(|| model.token_to_id(token))
    }

    /// Get the token matching the given id if it exists
    #[deprecated(
        since = "0.19.0",
        note = "please use `added_vocabulary.simple_id_to_token(id).or_else(|| model.id_to_token(id)` instead"
    )]
    pub fn id_to_token(&self, id: u32, model: &impl Model) -> Option<String> {
        self.added_tokens_map_r
            .get(&id)
            .map(|t| t.content.clone())
            .or_else(|| model.id_to_token(id))
    }

    pub fn simple_id_to_token(&self, id: u32) -> Option<String> {
        self.added_tokens_map_r.get(&id).map(|t| t.content.clone())
    }

    pub fn set_encode_special_tokens(&mut self, value: bool) {
        self.encode_special_tokens = value;
    }

    pub fn get_encode_special_tokens(&self) -> bool {
        self.encode_special_tokens
    }

    /// Check if a token is a special token
    pub fn is_special_token(&self, token: &str) -> bool {
        self.special_tokens_set.contains(token)
    }

    /// Add some special tokens to the vocabulary.
    /// The `normalizer` parameter is no longer used, as we are only handling non-normalized tokens.
    pub fn add_special_tokens(
        &mut self,
        tokens: &[AddedToken],
        model: &impl Model,
        // Removed: normalizer: Option<&N>,
    ) -> usize {
        self.add_tokens(tokens, model)
    }

    /// Add some tokens to the vocabulary.
    /// The `normalizer` parameter is no longer used, as we are only handling non-normalized tokens.
    pub fn add_tokens(
        &mut self,
        tokens: &[AddedToken],
        model: &impl Model,
        // Removed: normalizer: Option<&N>,
    ) -> usize {
        // Handle special tokens (if any)
        for token in tokens {
            if token.special
                && !token.content.is_empty()
                && !self.special_tokens_set.contains(&token.content)
            {
                self.special_tokens.push(token.to_owned());
                self.special_tokens_set.insert(token.content.clone());
            }
        }

        let mut ignored = 0;
        for token in tokens {
            if token.content.is_empty() || self.added_tokens_map_r.values().any(|val| val == token)
            {
                ignored += 1;
                continue;
            }
            // If a token is already part of the vocabulary, we mark it as added
            let new_id = if let Some(new_id) = self.token_to_id(&token.content, model) {
                new_id
            } else {
                self.added_tokens_map.values().cloned().max().map_or(
                    model.get_vocab_size() as u32,
                    |max| {
                        if (max >= model.get_vocab_size() as u32) || model.get_vocab_size() == 0 {
                            max + 1
                        } else {
                            model.get_vocab_size() as u32
                        }
                    },
                )
            };
            // Make sure we modify the previous entry
            self.added_tokens_map
                .entry(token.content.clone())
                .and_modify(|old_id| *old_id = new_id)
                .or_insert_with(|| new_id);
            // Update the current revert operation
            self.added_tokens_map_r
                .entry(new_id)
                .and_modify(|t| *t = token.clone())
                .or_insert_with(|| token.clone());
            // Make sure to remove previous entry (if the token gets a new id)

            // Finally add the token to the classic set if special
            if !self.special_tokens_set.contains(&token.content) {
                self.added_tokens.push(token.clone());
            }
        }

        // Call the simplified refresh_added_tokens
        self.refresh_added_tokens(model);

        // Return the number of added tokens
        tokens.len() - ignored
    }

    /// Reconstruct our internal AhoCorasick automaton when new tokens are added to the vocabulary.
    /// This now only handles non-normalized tokens.
    fn refresh_added_tokens(&mut self, model: &impl Model) {
        type TupleTokenId<'a> = (&'a AddedToken, u32);

        // Filter for only non-normalized tokens (since we're specializing)
        // Note: For special tokens, `normalized` is explicitly set to `false` in `AddedToken::from` and `Default`.
        let non_normalized: Vec<TupleTokenId> = self
            .special_tokens
            .iter()
            .chain(self.added_tokens.iter())
            .filter(|token| !token.normalized) // Explicitly filter
            .map(|token| {
                (
                    token,
                    self.token_to_id(&token.content, model)
                        .expect("Missing additional token"),
                )
            })
            .collect();

        let (tokens, ids): (Vec<&AddedToken>, Vec<u32>) = non_normalized.into_iter().unzip();
        let trie = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(tokens.iter().map(|token| &token.content))
            .expect("Failed to build trie when refreshing tokens");
        self.split_trie = (trie, ids);

        // Removed: Logic for normalized_trie
    }

    /// Find any AddedToken in the given sentence.
    /// This method returns a list of `Token`s that represent the matched special tokens,
    /// along with their original byte offsets.
    pub fn extract_special_tokens(&self, sentence: &str) -> Vec<Token> {
        if sentence.is_empty() {
            return vec![];
        }

        let mut extracted_tokens = Vec::new();

        for mat in self.split_trie.0.find_iter(sentence) {
            let mut start = mat.start();
            let mut stop = mat.end();
            let aho_id = mat.pattern();
            let id = self.split_trie.1[aho_id];
            let added_token = &self.added_tokens_map_r.get(&id).unwrap();

            // If special tokens should be ignored during encoding, skip them.
            if self.encode_special_tokens && self.special_tokens_set.contains(&added_token.content)
            {
                continue;
            }

            // Apply single_word constraint
            if added_token.single_word {
                let start_space = start == 0 || !ends_with_word(&sentence[..start]);
                let stop_space = stop == sentence.len() || !starts_with_word(&sentence[stop..]);

                if !stop_space || !start_space {
                    // Discard if it's not a single word match as per config
                    continue;
                }
            }

            // Apply lstrip/rstrip for offset adjustments
            let original_start = start; // Keep original start for lstrip logic
            if added_token.lstrip {
                let new_start_candidate = space_leftmost_at_end(&sentence[..start]);
                start = new_start_candidate; // Adjust start to include leading whitespace
            }
            if added_token.rstrip {
                stop += space_rightmost_at_start(&sentence[stop..]); // Adjust stop to include trailing whitespace
            }

            // Ensure adjusted offsets are valid and within the original bounds of the match
            start = std::cmp::min(start, original_start); // Ensure start doesn't go past the actual match start
            stop = std::cmp::max(stop, mat.end()); // Ensure stop doesn't go before the actual match end

            // The content of the token should be the actual string from the input,
            // with adjusted offsets for stripping if applicable.
            let token_content = sentence[start..stop].to_owned();
            extracted_tokens.push(Token::new(id, token_content, (start, stop)));
        }

        extracted_tokens
    }

    // Removed: split_with_indices
    // Removed: extract_and_normalize
}

impl Default for AddedVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct AddedTokenWithId {
    /// The id assigned to this token
    pub id: u32,
    #[serde(flatten)]
    /// The target AddedToken
    pub token: AddedToken,
}

impl Serialize for AddedVocabulary {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut added_tokens = self
            .added_tokens_map_r
            .iter()
            .map(|(id, token)| AddedTokenWithId {
                id: *id,
                token: token.clone(),
            })
            .collect::<Vec<_>>();
        // We need to have these added tokens ordered by ascending ID
        added_tokens.sort_unstable_by_key(|o| o.id);

        let mut vocabulary = serializer.serialize_seq(Some(added_tokens.len()))?;
        for token in added_tokens {
            vocabulary.serialize_element(&token)?;
        }

        vocabulary.end()
    }
}

