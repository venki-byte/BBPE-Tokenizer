// main.rs
mod tokenizer;

use tokenizer::Trainer;
use tokenizer::pre_tokenizer::{ByteLevel, PreTokenizedString, PreTokenizer};
use tokenizer::bpe_trainer::BpeTrainer;
use tokenizer::added_vocabulary::AddedToken;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write, BufRead};
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use tokenizer::Result; // Import your custom Result
use serde_json::{json, Value, Map};
// --- Data Structures for your JSON input ---
#[derive(Debug, Serialize, Deserialize)]
#[derive(Clone)]
struct DataEntry {
    text: String,
}

fn main() -> Result<()> {
    // --- Configuration ---
    let dataset_path = PathBuf::from("src/dataset/full_datas.jsonl");
    let output_dir = PathBuf::from("pretokenized_output");
    let output_file_path = output_dir.join("outputs.txt");
    let vocab_path = output_dir.join("vocab.txt");
    let merges_path = output_dir.join("merges.txt");
    let end_of_text_token = "<|endoftext|>".to_string();

    // Removed initial print statements for cleaner output during execution
    // println!("Starting pretokenization process...");
    // println!("Dataset: {:?}", dataset_path);
    // println!("Output directory: {:?}", output_dir);

    // --- 1. Load Data ---
    let file = fs::File::open(&dataset_path)?;
    let reader = std::io::BufReader::new(file);
    let data_entries: Vec<DataEntry> = reader
        .lines()
        .filter_map(|line_result| {
            line_result.ok().and_then(|line| serde_json::from_str::<DataEntry>(&line).ok())
        })
        .collect();
    // println!("Loaded {} data entries.", data_entries.len()); // Removed this print statement

    // --- 2. Initialize Pretokenizer ---
    let byte_level_pretokenizer = ByteLevel::new();
    fs::create_dir_all(&output_dir)?;

    // --- 3. Pretokenization ---
    let mut pretokenized_texts: Vec<String> = Vec::new();
    for (i, entry) in data_entries.iter().enumerate() {
        let mut pretokenized_string = PreTokenizedString::new(&entry.text);
        byte_level_pretokenizer.pre_tokenize(&mut pretokenized_string)
            .map_err(|e| tokenizer::Error::from(e.to_string()))?; // Using map_err for cleaner error conversion
        let processed_text = pretokenized_string
            .take_splits()
            .into_iter()
            .map(|ns| ns.get().to_string())
            .collect::<Vec<String>>()
            .join("");
        pretokenized_texts.push(processed_text);

        // Removed the periodic print statement, only keep if it's for user feedback on very long runs
        // if (i + 1) % 100 == 0 || (i + 1) == data_entries.len() {
        //     println!("Processed {} entries...", i + 1);
        // }
    }

    // --- 4. Train with BPETrainer ---
    let special_tokens = vec![
        AddedToken {
            content: end_of_text_token.clone(),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
            special: true,
        },
    ];

    let mut trainer = BpeTrainer::builder()
        .vocab_size(4000)
        .min_frequency(1000)
        .show_progress(true) // Kept show_progress as it's a trainer setting, not a debug print
        .special_tokens(special_tokens)
        //.continuing_subword_prefix("##".to_string()) // Uncomment if needed
        //.end_of_word_suffix("</w>".to_string()) // Uncomment if needed
        .build();

    trainer
        .feed(pretokenized_texts.iter().map(|s| s.as_str()), |s| {
            Ok(s.split_whitespace().map(|word| word.to_string()).collect::<Vec<String>>())
        })?;

    // FIX: Clone `trainer.words` before passing it to `do_train`
    // Ensure that `trainer.words` is actually exposed publicly or via a getter in BpeTrainer
    // This line might need adjustment based on how BpeTrainer is designed.
    // Assuming `trainer.words` is accessible and the cloning is necessary for `do_train`.
    let words_for_training = trainer.words.clone();
    let output = trainer.do_train(&words_for_training)?;


    
    // println!("Saved merges to {:?}", merges_path); // Removed print statement

    // 5. Write vocab.json
    let vocab_map = output.get_vocab();
    let mut ordered_vocab = Map::new();

    // 1) Single‐byte tokens in byte order
    for b in 0u8..=255 {
        let ch = char::from_u32(b as u32).unwrap();
        let key = ch.to_string();
        if let Some(&id) = vocab_map.get(&key) {
            ordered_vocab.insert(key, Value::from(id));
        }
    }

    // 2) Then merged tokens in ascending ID order
    let mut rest: Vec<_> = vocab_map
        .iter()
        .filter(|(tok, _)| tok.len() > 1)
        .collect();
    rest.sort_by_key(|&(_, &id)| id);
    for (token, &id) in rest {
        ordered_vocab.insert(token.clone(), Value::from(id));
    }

    // 3) Wrap under "vocab" and serialize
    let mut root = Map::new();
    root.insert("vocab".to_string(), Value::Object(ordered_vocab));
    let out = serde_json::to_string_pretty(&Value::Object(root))?;
    std::fs::write(output_dir.join("vocab.json"), out)?;

    // 6. Write merges.json (unchanged)
    let merges_json = json!({ "merges": output.get_merges() });
    let merges_file = File::create(output_dir.join("merges.json"))?;
    let mut merges_writer = BufWriter::new(merges_file);
    serde_json::to_writer_pretty(&mut merges_writer, &merges_json)?;
    merges_writer.flush()?;

    // 7. Save pretokenized outputs…
    Ok(())
}
