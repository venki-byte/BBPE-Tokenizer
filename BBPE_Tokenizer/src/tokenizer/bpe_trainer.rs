// src/tokenizer/bpe_trainer.rs
#![allow(clippy::map_entry)]
use crate::tokenizer::pair::Pair;
use crate::tokenizer::word::Word;
use rayon::iter::{IntoParallelRefMutIterator, IndexedParallelIterator, ParallelIterator};
use crate::tokenizer::parallelism::{MaybeParallelBridge, MaybeParallelRefIterator};
use crate::tokenizer::{AddedToken, Result, Trainer};
use crate::tokenizer::progress::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::File;
use std::io::BufWriter;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use crate::tokenizer::bpe::BPE;
use crate::tokenizer::pre_tokenizer::RE; // Used for tokenizer_pattern later

#[derive(Debug, Eq)]
struct Merge {
    pair: Pair,
    count: u64,
    merged_len: usize, // Length of the token string resulting from this merge
    merged_string: String, // NEW: Store the resulting string from the merge
}

impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count 
        && self.pair == other.pair 
        && self.merged_len == other.merged_len
        && self.merged_string == other.merged_string // NEW: Include string in PartialEq
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        // 1. Primary sort: By frequency (count) in DESCENDING order (max-heap behavior)
        self.count.cmp(&other.count)
            // 2. Secondary sort (if counts are equal): By merged_len in ASCENDING order (shorter first)
            .then_with(|| other.merged_len.cmp(&self.merged_len))
            // 3. Tertiary sort (if counts and lengths are equal): By merged_string in ASCENDING alphabetical order
            .then_with(|| other.merged_string.cmp(&other.merged_string)) // Corrected: self vs other
    }
}

struct Config {
    min_frequency: u64,
    vocab_size: usize,
    show_progress: bool,
    special_tokens: Vec<AddedToken>,
    limit_alphabet: Option<usize>,
    initial_alphabet: HashSet<char>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    max_token_length: Option<usize>,
    pub output_file: Option<String>,
}

pub struct BpeTrainerBuilder {
    config: Config,
}

impl Default for BpeTrainerBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                min_frequency: 1,
                vocab_size: 800,
                show_progress: true,
                special_tokens: vec![],
                limit_alphabet: None,
                initial_alphabet: HashSet::new(),
                continuing_subword_prefix: None,
                end_of_word_suffix: None,
                max_token_length: None,
                output_file: None, // Default to None, can be set later
            },
        }
    }
}

impl BpeTrainerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        self.config.min_frequency = frequency;
        self
    }

    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    #[must_use]
    pub fn show_progress(mut self, show: bool) -> Self {
        self.config.show_progress = show;
        self
    }

    #[must_use]
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        self.config.special_tokens = tokens;
        self
    }

    #[must_use]
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.config.limit_alphabet = Some(limit);
        self
    }

    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        self.config.initial_alphabet = alphabet;
        self
    }

    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.config.continuing_subword_prefix = Some(prefix);
        self
    }

    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        self.config.end_of_word_suffix = Some(suffix);
        self
    }

    #[must_use]
    pub fn max_token_length(mut self, max_token_length: Option<usize>) -> Self {
        self.config.max_token_length = max_token_length;
        self
    }

    pub fn output_path(mut self, path: String) -> Self {
        //self.config.output_path = Some(path);
        self.config.output_file = Some(path);
        self
    }


    pub fn build(self) -> BpeTrainer {
        BpeTrainer {
            min_frequency: self.config.min_frequency,
            vocab_size: self.config.vocab_size,
            show_progress: self.config.show_progress,
            special_tokens: self.config.special_tokens,
            limit_alphabet: self.config.limit_alphabet,
            initial_alphabet: self.config.initial_alphabet,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            end_of_word_suffix: self.config.end_of_word_suffix,
            max_token_length: self.config.max_token_length,
            output_file: self.config.output_file.clone(),
            words: HashMap::new(),
            special_token_ids: HashSet::new(),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
pub struct BpeTrainer {
    pub min_frequency: u64,
    pub output_file: Option<String>, // Optional output file path
    pub vocab_size: usize,
    pub show_progress: bool,
    pub special_tokens: Vec<AddedToken>,
    pub limit_alphabet: Option<usize>,
    pub initial_alphabet: HashSet<char>,
    pub continuing_subword_prefix: Option<String>,
    pub end_of_word_suffix: Option<String>,
    pub max_token_length: Option<usize>,

    pub words: HashMap<String, u64>,
    pub special_token_ids: HashSet<u32>, // Store IDs of actual special tokens (e.g., <unk>, <s>)
}

impl Default for BpeTrainer {
    fn default() -> Self {
        Self::builder().build()
    }
}

// Helper function to generate the GPT-2 byte-to-unicode mapping
fn bytes_to_unicode_map() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = Vec::new(); // byte values
    let mut cs: Vec<char> = Vec::new(); // corresponding unicode chars

    // Printable ASCII characters
    for i in b'!'..=b'~' { bs.push(i); cs.push(i as char); }

    // Some common extended ASCII characters (from GPT-2 tokenizer)
    for i in 0xA1..=0xAC { bs.push(i); cs.push(char::from_u32(i as u32).expect("Invalid Unicode for 0xA1-0xAC")); }
    for i in 0xAE..=0xFF { bs.push(i); cs.push(char::from_u32(i as u32).expect("Invalid Unicode for 0xAE-0xFF")); }

    // Now, handle all other bytes (control characters, whitespace, etc.)
    let mut n = 0x100; // Starting Unicode code point for remapped bytes
    for b in 0u8..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(char::from_u32(n).expect("Invalid Unicode for remapped bytes"));
            n += 1;
        }
    }

    bs.into_iter().zip(cs.into_iter()).collect()
}


impl BpeTrainer {
    pub fn new(min_frequency: u64, vocab_size: usize) -> Self {
        Self {
            min_frequency,
            vocab_size,
            ..Default::default()
        }
    }

    pub fn builder() -> BpeTrainerBuilder {
        BpeTrainerBuilder::new()
    }

    fn setup_progress(&self) -> Option<ProgressBar> {
        if self.show_progress {
            let p = ProgressBar::new(0);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {pos:>9!}/{len:<9!}")
                    .expect("Invalid progress template. This is a hardcoded internal error, should not happen."),
            );
            Some(p)
        } else {
            None
        }
    }

    fn finalize_progress(&self, p: &Option<ProgressBar>, final_len: usize) {
        if let Some(p) = p {
            p.set_length(final_len as u64);
            p.finish();
        }
    }

    fn update_progress(&self, p: &Option<ProgressBar>, len: usize, message: &'static str) {
        if let Some(p) = p {
            p.set_message(message);
            p.set_length(len as u64);
            p.reset();
        }
    }

    fn add_special_tokens(
        &mut self,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
    ) {
        for token in &self.special_tokens {
            if !w2id.contains_key(&token.content) {
                id2w.push(token.content.to_owned());
                let id = (id2w.len() - 1) as u32;
                w2id.insert(token.content.to_owned(), id);
                self.special_token_ids.insert(id);
            }
        }
    }

    fn compute_alphabet(
        &self,
        wc: &HashMap<String, u64>,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
    ) {
        let byte_map = bytes_to_unicode_map();
        
        let initial_byte_chars: Vec<String> = (0u8..=255u8)
            .map(|i| byte_map[&i].to_string())
            .collect();

        // no sort here—preserves byte order 0..255
        for s in initial_byte_chars {
            if !w2id.contains_key(&s) {
                id2w.push(s.clone());
                w2id.insert(s.clone(), (id2w.len() - 1) as u32);
            }
        }

        let space_prefix_char = 'Ġ';
        let newline_char = 'Ċ';

        if !w2id.contains_key(&space_prefix_char.to_string()) && !self.special_tokens.iter().any(|t| t.content == space_prefix_char.to_string()) {
            id2w.push(space_prefix_char.to_string());
            w2id.insert(space_prefix_char.to_string(), (id2w.len() - 1) as u32);
        }
        if !w2id.contains_key(&newline_char.to_string()) && !self.special_tokens.iter().any(|t| t.content == newline_char.to_string()) {
            id2w.push(newline_char.to_string());
            w2id.insert(newline_char.to_string(), (id2w.len() - 1) as u32);
        }

        let mut additional_alphabet_chars: Vec<char> = self.initial_alphabet.iter().cloned().collect();
        additional_alphabet_chars.sort_unstable_by_key(|&c| c as u32);
        for c in additional_alphabet_chars {
            let s = c.to_string();
            if !w2id.contains_key(&s) { 
                id2w.push(s.clone());
                w2id.insert(s.clone(), (id2w.len() - 1) as u32);
            }
        }

        let mut other_chars_from_wc: HashMap<char, usize> = HashMap::new();
        for (word, count) in wc {
            for c in word.chars() {
                if !w2id.contains_key(&c.to_string()) {
                    other_chars_from_wc
                        .entry(c)
                        .and_modify(|cnt| *cnt += *count as usize)
                        .or_insert(*count as usize);
                }
            }
        }

        let mut kept_dynamic_chars: Vec<_> = other_chars_from_wc.iter().collect();
        let to_remove = self
            .limit_alphabet
            .map(|limit| {
                if kept_dynamic_chars.len() > limit {
                    kept_dynamic_chars.len() - limit
                } else {
                    0
                }
            })
            .unwrap_or(0);

        if to_remove > 0 {
            kept_dynamic_chars.sort_unstable_by_key(|k| *k.1);
            kept_dynamic_chars.drain(..to_remove);
        }

        kept_dynamic_chars.sort_unstable_by_key(|k| (*k.0) as u32);
        for (c, _) in kept_dynamic_chars {
            let s = c.to_string();
            if !w2id.contains_key(&s) { 
                id2w.push(s.clone());
                w2id.insert(s.clone(), (id2w.len() - 1) as u32);
            }
        }
    }

    fn tokenize_words(
        &self,
        wc: &HashMap<String, u64>,
        w2id: &mut HashMap<String, u32>,
        id2w: &Vec<String>,
        p: &Option<ProgressBar>,
    ) -> (Vec<Word>, Vec<u64>) {
        let mut words: Vec<Word> = Vec::with_capacity(wc.len());
        let mut counts: Vec<u64> = Vec::with_capacity(wc.len());

        for (word_str, count) in wc {
            let mut current_word = Word::new();
            counts.push(*count);
            let mut is_special_token_word = false;

            for special_token in &self.special_tokens {
                if word_str == &special_token.content {
                    if let Some(&id) = w2id.get(word_str.as_str()) {
                        current_word.add(id, word_str.len());
                        is_special_token_word = true;
                        break;
                    }
                }
            }

            if !is_special_token_word {
                for c in word_str.chars() {
                    let s = c.to_string();
                    if let Some(&id) = w2id.get(&s) {
                        current_word.add(id, s.len());
                    } else {
                        panic!("BUG: Character '{}' from word '{}' not found in vocabulary during tokenize_words. This indicates an issue with `compute_alphabet` or input data.", s, word_str);
                    }
                }
            }
            words.push(current_word);

            if let Some(p) = p {
                p.inc(1);
            }
        }
        (words, counts)
    }

    fn count_pairs(
        &self,
        words: &[Word],
        counts: &[u64],
        p: &Option<ProgressBar>,
    ) -> (HashMap<Pair, i32>, HashMap<Pair, HashSet<usize>>) {
        words
            .maybe_par_iter()
            .enumerate()
            .map(|(i, word)| {
                let mut pair_counts_local = HashMap::new();
                let mut where_to_update_local: HashMap<Pair, HashSet<usize>> = HashMap::new();

                for window in word.get_chars().windows(2) {
                    let cur_pair: Pair = (window[0], window[1]);

                    let count = counts[i];
                    *pair_counts_local.entry(cur_pair).or_insert(0) += count as i32;

                    where_to_update_local
                        .entry(cur_pair)
                        .or_insert_with(HashSet::new)
                        .insert(i);
                }

                if let Some(p) = &p {
                    p.inc(1);
                }

                (pair_counts_local, where_to_update_local)
            })
            .reduce(
                || (HashMap::new(), HashMap::new()),
                |(mut pair_counts_global, mut where_to_update_global), (pc_local, wtu_local)| {
                    for (k, v) in pc_local {
                        *pair_counts_global.entry(k).or_insert(0) += v;
                    }
                    for (k, v) in wtu_local {
                        where_to_update_global
                            .entry(k)
                            .or_insert_with(HashSet::new)
                            .extend(v);
                    }
                    (pair_counts_global, where_to_update_global)
                },
            )
    }

    pub fn do_train(
        &mut self,
        word_counts: &HashMap<String, u64>,
    ) -> Result<BPE> {
        let progress = self.setup_progress();
        let mut word_to_id: HashMap<String, u32> = HashMap::new();
        let mut id2w: Vec<String> = Vec::new();

        self.compute_alphabet(word_counts, &mut word_to_id, &mut id2w);
        
        let max_token_length = self.max_token_length.unwrap_or(usize::MAX);

        self.update_progress(&progress, word_counts.len(), "Tokenizing words");
        let (mut words, counts) =
            self.tokenize_words(word_counts, &mut word_to_id, &id2w, &progress);
        self.finalize_progress(&progress, words.len());

        self.update_progress(&progress, words.len(), "Count pairs");
        let (mut pair_counts, mut where_to_update) = self.count_pairs(&words, &counts, &progress);
        
        let mut queue = BinaryHeap::with_capacity(pair_counts.len());

        for (&pair, &count) in pair_counts.iter() {
            let part_a_string = id2w[pair.0 as usize].clone();
            let part_b_string = id2w[pair.1 as usize].clone();
            let new_token_string = format!("{part_a_string}{part_b_string}");

            let a_is_single_char_digit = part_a_string.len() == 1 && part_a_string.chars().next().map_or(false, |c| c.is_ascii_digit());
            let b_is_single_char_digit = part_b_string.len() == 1 && part_b_string.chars().next().map_or(false, |c| c.is_ascii_digit());

            let is_special_token_merge = self.special_token_ids.contains(&pair.0) || self.special_token_ids.contains(&pair.1);
            
            let is_forbidden_newline_merge = 
                (part_a_string == "Ċ" && part_b_string != "Ċ") || 
                (part_a_string != "Ċ" && part_b_string == "Ċ");    

            let is_forbidden_space_merge = 
                (part_a_string != "Ġ" && part_b_string == "Ġ") || 
                (part_a_string != "Ġ" && part_b_string.starts_with("Ġ") && part_b_string != "Ġ"); 

            let is_forbidden_digit_merge = a_is_single_char_digit && b_is_single_char_digit;

            let should_skip = is_special_token_merge || is_forbidden_newline_merge || is_forbidden_space_merge || is_forbidden_digit_merge;

            if should_skip {
                continue;
            }
            let merged_len = part_a_string.len() + part_b_string.len();

            if count >= self.min_frequency as i32 {
                queue.push(Merge {
                    pair,
                    count: count as u64,
                    merged_len,
                    merged_string: new_token_string,
                });
            }
        }
        self.finalize_progress(&progress, words.len());
        self.update_progress(&progress, self.vocab_size as usize, "Compute merges");

        let mut final_merges_raw_pairs: Vec<(u32, u32)> = vec![];
        let special_token_ids_for_merge_prevention = &self.special_token_ids;

        loop {
            if word_to_id.len() >= self.vocab_size {
                break;
            }
            if queue.is_empty() {
                break;
            }

            let top = queue.pop().expect("Queue unexpectedly empty after check. This indicates a logic error.");
            
            let current_pair_count = pair_counts.get(&top.pair).copied().unwrap_or(0);
            
            if current_pair_count == 0 || top.count > current_pair_count as u64 {
                if let Some(p) = &progress {
                    p.inc(1);
                }
                continue;
            }

            if current_pair_count < self.min_frequency as i32 {
                if let Some(p) = &progress {
                    p.inc(1);
                }
                continue;
            }

            let part_a_string = id2w[top.pair.0 as usize].clone();
            let part_b_string = id2w[top.pair.1 as usize].clone();

            let a_is_single_char_digit = part_a_string.len() == 1
                && part_a_string.chars().next().map_or(false, |c| c.is_ascii_digit());
            let b_is_single_char_digit = part_b_string.len() == 1
                && part_b_string.chars().next().map_or(false, |c| c.is_ascii_digit());

            let is_special_token_merge = special_token_ids_for_merge_prevention.contains(&top.pair.0) || special_token_ids_for_merge_prevention.contains(&top.pair.1);
            
            let is_forbidden_newline_merge = 
                (part_a_string == "Ċ" && part_b_string != "Ċ") || 
                (part_a_string != "Ċ" && part_b_string == "Ċ");    
            let is_forbidden_space_merge = 
                (part_a_string != "Ġ" && part_b_string == "Ġ") || 
                (part_a_string != "Ġ" && part_b_string.starts_with("Ġ") && part_b_string != "Ġ"); 

            let is_forbidden_digit_merge = a_is_single_char_digit && b_is_single_char_digit;

            let should_skip = is_special_token_merge || is_forbidden_newline_merge || is_forbidden_space_merge || is_forbidden_digit_merge;

            if should_skip {
                if let Some(p) = &progress { p.inc(1); }
                continue;
            }
            
            let new_token_string = format!("{part_a_string}{part_b_string}");
            if new_token_string.len() > max_token_length {
                if let Some(p) = &progress {
                    p.inc(1);
                }
                continue;
            }
            let new_token_id = id2w.len() as u32;
            id2w.push(new_token_string.clone());
            word_to_id.insert(new_token_string.clone(), new_token_id);

            final_merges_raw_pairs.push(top.pair);

            let words_to_process_indices: HashSet<usize> = where_to_update
                .get(&top.pair)
                .cloned()
                .unwrap_or_default();
            
            let collected_changes = words
                .par_iter_mut()
                .enumerate()
                .filter_map(|(i, word)| {
                    if words_to_process_indices.contains(&i) {
                        let merge_result = word.merge(top.pair.0, top.pair.1, new_token_id, max_token_length, &id2w);
                        Some(
                            merge_result.into_iter()
                                .map(|c| (c, i))
                                .collect::<Vec<_>>(),
                        )
                    } else {
                        None
                    }
                })
                .flatten()
                .collect::<Vec<_>>();

            if let Some(p) = &progress {
                p.inc(1);
            }
            let mut pairs_to_re_evaluate: HashSet<Pair> = HashSet::new();
            for ((pair, change_type), word_idx) in collected_changes {
                let count_change = change_type * counts[word_idx] as i32;
                if count_change != 0 {
                    let entry = pair_counts.entry(pair);
                    match entry {
                        std::collections::hash_map::Entry::Occupied(mut occ) => {
                            *occ.get_mut() += count_change;
                            if *occ.get() <= 0 {
                                occ.remove();
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(vac) => {
                            if count_change > 0 {
                                vac.insert(count_change);
                            }
                        }
                    }
                    pairs_to_re_evaluate.insert(pair);
                }
                if change_type == -1 {
                    if let Some(indices) = where_to_update.get_mut(&pair) {
                        indices.remove(&word_idx);
                        if indices.is_empty() {
                            where_to_update.remove(&pair);
                        }
                    }
                } else {
                    where_to_update
                        .entry(pair)
                        .or_insert_with(HashSet::new)
                        .insert(word_idx);
                }
            }
            
            for &pair_key in &pairs_to_re_evaluate {
                if let Some(&count) = pair_counts.get(&pair_key) {
                    let part_a_string = id2w.get(pair_key.0 as usize).expect("ID in pair_key not found in id2w during re-evaluation. This indicates a logic error.");
                    let part_b_string = id2w.get(pair_key.1 as usize).expect("ID in pair_key not found in id2w during re-evaluation. This indicates a logic error.");
                    let re_evaluated_new_token_string = format!("{part_a_string}{part_b_string}"); // Re-calculate merged string

                    let a_is_single_char_digit = part_a_string.len() == 1
                        && part_a_string.chars().next().map_or(false, |c| c.is_ascii_digit());
                    let b_is_single_char_digit = part_b_string.len() == 1
                        && part_b_string.chars().next().map_or(false, |c| c.is_ascii_digit());

                    let is_special_token_merge = special_token_ids_for_merge_prevention.contains(&pair_key.0) || special_token_ids_for_merge_prevention.contains(&pair_key.1);
                    
                    let is_forbidden_newline_merge = 
                        (part_a_string == "Ċ" && part_b_string != "Ċ") || 
                        (part_a_string != "Ċ" && part_b_string == "Ċ");    

                    let is_forbidden_space_merge = 
                        (part_a_string != "Ġ" && part_b_string == "Ġ") || 
                        (part_a_string != "Ġ" && part_b_string.starts_with("Ġ") && part_b_string != "Ġ"); 

                    let is_forbidden_digit_merge = a_is_single_char_digit && b_is_single_char_digit;

                    let should_skip = is_special_token_merge || is_forbidden_newline_merge || is_forbidden_space_merge || is_forbidden_digit_merge;

                    if should_skip {
                        continue;
                    }
                    let merged_len = part_a_string.len() + part_b_string.len();

                    if count >= self.min_frequency as i32 {
                        queue.push(Merge {
                            pair: pair_key,
                            count: count as u64,
                            merged_len,
                            merged_string: re_evaluated_new_token_string,
                        });
                    } else {
                        // This case was implicitly handled by the previous debug print being removed.
                        // No need for a print here; the item simply isn't re-pushed.
                    }
                } else {
                    // This case was implicitly handled by the previous debug print being removed.
                    // No need for a print here; the item simply isn't re-pushed.
                }
            }
            if let Some(p) = &progress {
                p.set_position(word_to_id.len() as u64);
            }
        }
        self.finalize_progress(&progress, word_to_id.len());
        self.add_special_tokens(&mut word_to_id, &mut id2w);

        let mut compact_word_to_id: HashMap<String, u32> = HashMap::new();
        let mut compact_id2w: Vec<String> = Vec::new();
        let mut old_to_new: HashMap<u32, u32> = HashMap::new();

        // --- DEBUG PRINTS (UNCOMMENT TO USE) ---
        println!("\n--- Debugging Compaction Start ---");
        println!("State before compaction loop:");
        println!("id2w length: {}", id2w.len());
        println!("word_to_id length: {}", word_to_id.len());
        println!("First 10 id2w entries: {:?}", id2w.iter().take(10).collect::<Vec<_>>());
        println!("First 10 word_to_id entries: {:?}", word_to_id.iter().take(10).collect::<Vec<_>>());
        println!("Number of raw merges: {}", final_merges_raw_pairs.len());
        
        // Find and print the problematic token BEFORE the loop
        let problematic_old_id = 1235; // Replace with the ID from your panic message
        if let Some(token_str_from_id2w) = id2w.get(problematic_old_id as usize) {
            println!("TOKEN FOR OLD_ID {}: '{}'", problematic_old_id, token_str_from_id2w);
            if word_to_id.contains_key(token_str_from_id2w) {
                println!("    -> This token string IS present in word_to_id.");
                println!("    -> Its ID in word_to_id: {:?}", word_to_id.get(token_str_from_id2w));
            } else {
                println!("    -> WARNING: This token string IS NOT present in word_to_id.");
            }
        } else {
            println!("ERROR: OLD_ID {} is out of bounds for id2w! id2w.len() = {}", problematic_old_id, id2w.len());
        }
        // --- END DEBUG PRINTS ---


        // Populate compact_word_to_id, compact_id2w, and old_to_new mapping
        // We iterate through the `id2w` that was built during training (which includes original tokens and new merged tokens).
        // For each of these, if it still exists in the final `word_to_id` (meaning it wasn't somehow removed),
        // we give it a new compact ID.
        for (old_id_idx, token_str) in id2w.iter().enumerate() {
            let old_id = old_id_idx as u32; // Cast to u32 for consistency with HashMap keys

            if let Some(&final_id_in_word_to_id) = word_to_id.get(token_str) {
                 // Check if `old_to_new` already has this old_id.
                 // This guards against duplicates if `word_to_id` had multiple entries mapping to the same token_str,
                 // which shouldn't happen with unique IDs, but defensive.
                if !old_to_new.contains_key(&old_id) {
                    if !compact_word_to_id.contains_key(token_str) {
                        let new_id = compact_id2w.len() as u32;
                        compact_word_to_id.insert(token_str.clone(), new_id);
                        compact_id2w.push(token_str.clone());
                        old_to_new.insert(old_id, new_id);

                        // --- DEBUG PRINT (UNCOMMENT TO USE) ---
                        // if old_id == problematic_old_id {
                        //     println!("DEBUG: MAPPED problematic old_id {}: '{}' -> new_id: {}", old_id, token_str, new_id);
                        // }
                        // --- END DEBUG PRINT ---
                    } else {
                        // This token string already exists in compact_word_to_id,
                        // so find its new ID and map the current old_id to it.
                        // This handles cases where different old_ids (e.g., from character-level vs merged-token)
                        // might eventually resolve to the same string content after some merges.
                        // However, with our current ID assignment (sequential), this is unlikely.
                        let existing_new_id = *compact_word_to_id.get(token_str).expect("Logic error: token should be in compact_word_to_id if contains_key is true");
                        old_to_new.insert(old_id, existing_new_id);

                        // --- DEBUG PRINT (UNCOMMENT TO USE) ---
                        // if old_id == problematic_old_id {
                        //     println!("DEBUG: MAPPED problematic old_id {} ('{}') to existing new_id {}", old_id, token_str, existing_new_id);
                        // }
                        // --- END DEBUG PRINT ---
                    }
                }
            } else {
                // This is the problematic path for old_id 1235.
                // If the token string from id2w is not in word_to_id, it means it's been "retired".
                // We *must* ensure that tokens used in final_merges_raw_pairs are never retired in this way.
                // --- DEBUG PRINT (UNCOMMENT TO USE) ---
                // if old_id == problematic_old_id {
                //     println!("DEBUG: Problematic old_id {} ('{}') was NOT found in final word_to_id. This is why it's not mapped.", old_id, token_str);
                // } else {
                //     // println!("DEBUG: Token '{}' (old_id: {}) from id2w not found in word_to_id during compaction prep (normal for retired tokens).", token_str, old_id);
                // }
                // --- END DEBUG PRINTS ---
            }
        }

        // --- DEBUG PRINTS (UNCOMMENT TO USE) ---
        let problematic_old_id = 1235; // This needs to be defined if you uncomment the block
        println!("\nOld to New ID map (after population, size {}): {:?}", old_to_new.len(), old_to_new.iter().map(|(&k, &v)| (k, v)).collect::<Vec<_>>().iter().take(20).collect::<Vec<_>>());
        if old_to_new.contains_key(&(problematic_old_id as u32)) {
            println!("CONFIRM: Problematic old_id {} IS in old_to_new map, mapping to {}", problematic_old_id, old_to_new[&(problematic_old_id as u32)]);
        } else {
            println!("CONFIRM: Problematic old_id {} IS NOT in old_to_new map. The reason should be evident from earlier debug prints.", problematic_old_id);
        }

        println!("Compact vocab size: {}", compact_word_to_id.len());
        println!("Compact id2w size: {}", compact_id2w.len());
        println!("--- Debugging Compaction End ---");
        // --- END DEBUG PRINTS ---

        let mut compact_special_tokens_encoder: HashMap<String, u32> = HashMap::new();
        for token in &self.special_tokens {
            if let Some(&new_id) = compact_word_to_id.get(&token.content) {
                compact_special_tokens_encoder.insert(token.content.clone(), new_id);
            } else {
                panic!("Special token '{}' not found in final compacted vocabulary. This should not happen if special tokens are correctly managed and not filtered out.", token.content);
            }
        }

        let compact_merges = final_merges_raw_pairs
            .iter()
            .filter_map(|(old_id_a, old_id_b)| {
                // The panic indicates one of these `.get()` calls failed.
                // If old_to_new was correctly populated, this shouldn't fail for IDs from final_merges_raw_pairs.
                // The error now provides the specific old ID.
                let new_a = old_to_new.get(old_id_a).expect(&format!("Old ID {} for merge part A not found in new mapping during compaction. This indicates a logic error.", old_id_a));
                let new_b = old_to_new.get(old_id_b).expect(&format!("Old ID {} for merge part B not found in new mapping during compaction. This indicates a logic error.", old_id_b));

                let s1 = compact_id2w.get(*new_a as usize).expect(&format!("New ID {} for merge part A not found in compact_id2w. This indicates a logic error.", new_a)).clone();
                let s2 = compact_id2w.get(*new_b as usize).expect(&format!("New ID {} for merge part B not found in compact_id2w. This indicates a logic error.", new_b)).clone();
                
                Some((s1, s2))
            })
            .collect::<Vec<_>>();

        let tokenizer_pattern = RE.as_str();

        let mut bpe = BPE::new(
            // Use compact_id2w to get the ordered vocabulary for BPE
            compact_id2w
                .iter()
                .enumerate()
                .map(|(id, token_str)| (token_str.as_bytes().to_vec(), id as u32))
                .collect::<Vec<(Vec<u8>, u32)>>(),
            compact_special_tokens_encoder,
            tokenizer_pattern,
        ).map_err(|e| crate::tokenizer::Error::from(e))?;

        // Construct a BTreeMap for ordered JSON output
        use std::collections::BTreeMap;
        let mut ordered_vocab = BTreeMap::new();
        for (id, token_str) in compact_id2w.iter().enumerate() {
            ordered_vocab.insert(token_str.clone(), id as u32);
        }

        let out = json!({
            "vocab" : ordered_vocab, // Use the BTreeMap for ordered output
            "merges" : compact_merges.clone(),
        });
        let path = self.output_file.as_deref().unwrap_or("tokenizer.json");
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &out)?;
        bpe.merges = compact_merges; // Ensure merges are set in the BPE object
        Ok(bpe)
    }
}

impl Trainer for BpeTrainer {
    type Model = BPE;

    fn train(&self, _model: &mut Self::Model) -> Result<Vec<AddedToken>> {
        let mut temp_self = self.clone();
        let words_copy = self.words.clone();
        *_model = temp_self.do_train(&words_copy)?;
        
        Ok(self.special_tokens.clone())
    }

    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        let words: Result<HashMap<String, u64>> = iterator
            .maybe_par_bridge()
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = HashMap::new();
                for word in words {
                    map.entry(word).and_modify(|c| *c += 1).or_insert(1);
                }
                Ok(map)
            })
            .reduce(
                || Ok(HashMap::new()),
                |acc, ws| {
                    let mut acc = acc?;
                    for (k, v) in ws? {
                        acc.entry(k).and_modify(|c| *c += v).or_insert(v);
                    }
                    Ok(acc)
                },
            );

        self.words = words?;
       
        Ok(())
    }
}