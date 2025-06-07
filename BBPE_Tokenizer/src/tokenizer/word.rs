// src/tokenizer/word.rs

use crate::tokenizer::pair::Pair;
use std::collections::HashMap; // Needed for id_to_word in merge
use std::ops::{Deref, DerefMut};
use std::borrow::Borrow;

#[cfg(feature = "random")]
use rand::{thread_rng, Rng};

/// Represents a single word, storing it as a vector of u32 (ids to the vocabulary)
/// and a vector of the original byte lengths of those tokens.
#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct Word {
    pub chars: Vec<u32>,
    pub char_byte_lengths: Vec<usize>,
    #[cfg(feature = "random")]
    pub dropout: Option<Vec<f32>>,
}

impl Word {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new token to the word. Store its ID and its byte length.
    pub fn add(&mut self, char_id: u32, char_len: usize) {
        self.chars.push(char_id);
        self.char_byte_lengths.push(char_len);
    }

    pub fn get_chars(&self) -> &[u32] {
        self.chars.borrow()
    }

    pub fn is_enabled(&self) -> bool {
        #[cfg(not(feature = "random"))]
        {
            true
        }
        #[cfg(feature = "random")]
        {
            if let Some(dropout_vec) = &self.dropout {
                let rand = thread_rng().r#gen::<f32>();
                rand < dropout_vec[0]
            } else {
                true
            }
        }
    }

    /// Perform a merge operation on this word.
    /// Needs `id_to_word` to calculate the length of the new token string.
    pub fn merge(
        &mut self,
        a: u32,
        b: u32,
        new_id: u32,
        max_token_length: usize,
        id_to_word: &Vec<String>, // Pass id_to_word here
    ) -> Vec<(Pair, i32)> {
        let mut changes = Vec::new();

        let mut i = 0;
        let mut new_chars = Vec::with_capacity(self.chars.len());
        let mut new_char_byte_lengths = Vec::with_capacity(self.char_byte_lengths.len());
        #[cfg(feature = "random")]
        let mut new_dropout = self.dropout.as_ref().map(|d| Vec::with_capacity(d.len()));

        while i < self.chars.len() {
            // Check for the pair (a, b) at the current position
            if i + 1 < self.chars.len() && self.chars[i] == a && self.chars[i + 1] == b {
                // Calculate the actual length of the potential new token string
                let token_a_str = &id_to_word[a as usize];
                let token_b_str = &id_to_word[b as usize];
                let new_token_len = token_a_str.len() + token_b_str.len(); // Calculate byte length

                if max_token_length != usize::MAX && new_token_len > max_token_length {
                    // This merge would create a token too long, skip this specific merge occurrence
                    // We just push the current char 'a' and move to the next position.
                    // The pair (a, b) at this specific location is not merged.
                    new_chars.push(self.chars[i]);
                    new_char_byte_lengths.push(self.char_byte_lengths[i]);
                    #[cfg(feature = "random")]
                    if let Some(d) = &mut new_dropout { d.push(self.dropout.as_ref().unwrap()[i]); }
                    i += 1; // Move to the next char, potentially starting a new pair check with 'b'
                    continue;
                }

                // --- Pair (a, b) is being merged into new_id ---

                // 1. Report old pairs that are removed:
                //    - The merged pair (a, b) itself
                changes.push(((a, b), -1));

                //    - The pair (previous_token, a) if it exists
                if !new_chars.is_empty() { // There was a token before 'a'
                    let previous_token = new_chars[new_chars.len() - 1];
                    changes.push(((previous_token, a), -1));
                }

                //    - The pair (b, next_token) if it exists
                if i + 2 < self.chars.len() { // There is a token after 'b'
                    let next_token = self.chars[i + 2];
                    changes.push(((b, next_token), -1));
                }

                // Add the new merged token
                new_chars.push(new_id);
                new_char_byte_lengths.push(new_token_len);
                #[cfg(feature = "random")]
                if let Some(d) = &mut new_dropout { d.push(self.dropout.as_ref().unwrap()[i]); } // Keep the dropout of the first char

                // 2. Report new pairs that are formed:
                //    - The pair (previous_token, new_id) if it exists
                if new_chars.len() >= 2 { // new_chars now contains previous_token and new_id
                    let previous_token = new_chars[new_chars.len() - 2];
                    changes.push(((previous_token, new_id), 1));
                }

                //    - The pair (new_id, next_token) if it exists
                if i + 2 < self.chars.len() { // next_token is at self.chars[i + 2] (original index)
                    let next_token = self.chars[i + 2];
                    changes.push(((new_id, next_token), 1));
                }

                i += 2; // Consume both parts of the pair (a, b)
            } else {
                // No merge at this position, just push the current char
                new_chars.push(self.chars[i]);
                new_char_byte_lengths.push(self.char_byte_lengths[i]);
                #[cfg(feature = "random")]
                if let Some(d) = &mut new_dropout { d.push(self.dropout.as_ref().unwrap()[i]); }
                i += 1;
            }
        }
        self.chars = new_chars;
        self.char_byte_lengths = new_char_byte_lengths;
        #[cfg(feature = "random")]
        {
            self.dropout = new_dropout;
        }

        changes
    }

    /// This function changes the `dropout` probability of the current word
    /// to 0, meaning it will always be enabled. It returns the dropout
    /// that has been removed, so that it can be reverted when needed.
    pub fn disable_dropout(&mut self) -> Option<Vec<f32>> {
        #[cfg(feature = "random")]
        {
            if let Some(mut dropout_vec) = self.dropout.take() {
                let old_dropout = dropout_vec[0];
                dropout_vec[0] = 0.0;
                self.dropout = Some(dropout_vec);
                Some(vec![old_dropout])
            } else {
                None
            }
        }
        #[cfg(not(feature = "random"))]
        {
            None
        }
    }

    /// This function sets the `dropout` to the given value, returning the old
    /// value if any.
    pub fn set_dropout(&mut self, dropout: Option<Vec<f32>>) -> Option<Vec<f32>> {
        #[cfg(feature = "random")]
        {
            self.dropout.replace(dropout.unwrap_or_else(|| vec![])) // Assume dropout is not None if this branch taken
        }
        #[cfg(not(feature = "random"))]
        {
            None
        }
    }
}