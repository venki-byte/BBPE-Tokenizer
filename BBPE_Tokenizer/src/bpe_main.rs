// src/bpe_main.rs

use std::{
    fs::{File, metadata},
    io::{self, Read, BufReader, BufRead, BufWriter, Write},
    collections::HashMap as StdHashMap,
    path::Path,
};
use rustc_hash::FxHashMap;
use serde_json::Value;
mod tokenizer;
use tokenizer::bpe::BPE;
use tokenizer::pre_tokenizer::{RE, ByteLevel, PreTokenizedString, PreTokenizer};

fn load_bpe_from_files(vocab_path: &str, merges_path: &str) -> BPE {

    // 1) Load and parse vocab.json
    let mut f = File::open(vocab_path).expect("vocab.json not found");
    let mut s = String::new();
    f.read_to_string(&mut s).unwrap();
    let v: Value = serde_json::from_str(&s).unwrap();
    let vocab_map = v["vocab"]
        .as_object().unwrap()
        .iter()
        .map(|(tok, id_v)| (tok.clone(), id_v.as_u64().unwrap() as u32))
        .collect::<StdHashMap<_,_>>();
 
    // 2) Build encoder (bytes→id) and special_tokens_encoder (string→id)
    let mut encoder: FxHashMap<Vec<u8>, u32>          = FxHashMap::default();
    let mut special_tokens_encoder: FxHashMap<String,u32> = FxHashMap::default();

    // Put *all* vocab entries into the encoder map:
    for (tok, &id) in &vocab_map {
        encoder.insert(tok.as_bytes().to_vec(), id);
    }

    // Then cherry-pick the real “special” tokens:
    if let Some(&eot_id) = vocab_map.get("<|endoftext|>") {
        special_tokens_encoder.insert("<|endoftext|>".to_string(), eot_id);
        // optionally remove from encoder so they're only matched as specials
        encoder.remove(&"<|endoftext|>".as_bytes().to_vec());
    }
 
    // 3) Load merges.json
    let mut f = File::open(merges_path).expect("merges.json not found");
    s.clear();
    f.read_to_string(&mut s).unwrap();
    let m: Value = serde_json::from_str(&s).unwrap();
    let merges = m["merges"]
        .as_array().unwrap()
        .iter()
        .map(|pair_v| {
            let arr = pair_v.as_array().unwrap();
            (arr[0].as_str().unwrap().to_string(),
             arr[1].as_str().unwrap().to_string())
        })
        .collect::<Vec<_>>();
 
    // 4) Construct BPE
    let mut bpe = BPE::new(encoder, special_tokens_encoder, RE.as_str())
        .expect("failed to create BPE");
    bpe.merges = merges;
    bpe

  
}

fn main() -> io::Result<()> {
    let out_dir       = Path::new("pretokenized_output");
    let vocab_path    = out_dir.join("vocab.json");
    let merges_path   = out_dir.join("merges.json");
    let test_path     = out_dir.join("test_data.txt");
    let encoded_path  = out_dir.join("encoded.txt");
    let decoded_path  = out_dir.join("decoded.txt");

    // load model
    let bpe = load_bpe_from_files(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
    );
    let allowed_special = bpe.special_tokens();

    // prepare pretokenizer & IO
    let byte_level = ByteLevel::new();
    let test_file   = File::open(&test_path)?;
    let mut reader  = BufReader::new(test_file);
    let mut enc_w   = BufWriter::new(File::create(&encoded_path)?);
    let mut dec_w   = BufWriter::new(File::create(&decoded_path)?);

    // counter for total tokens
    let mut total_tokens = 0usize;
    let mut raw_line     = String::new();

    while reader.read_line(&mut raw_line)? > 0 {
        let line = raw_line.trim_end_matches(&['\r','\n'][..]);

        // 1) pretokenize
        let mut pts = PreTokenizedString::new(line);
        byte_level
            .pre_tokenize(&mut pts)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        let pre = pts.take_splits()
                     .into_iter()
                     .map(|ns| ns.get().to_string())
                     .collect::<Vec<_>>()
                     .join("");

        // 2) encode
        let (tokens, _) = bpe.encode(&pre, &allowed_special);
        total_tokens  += tokens.len();
        let json_arr   = serde_json::to_string(&tokens).unwrap();
        writeln!(enc_w, "{}", json_arr)?;

        // 3) decode & replace Ġ → space
        let bytes      = bpe.decode_bytes(&tokens).unwrap();
        let text       = String::from_utf8(bytes).unwrap();
        let cleaned    = text.replace('Ġ', " ");
        writeln!(dec_w, "{}", cleaned)?;

        raw_line.clear();
    }

    enc_w.flush()?;
    dec_w.flush()?;

    println!("total tokens written to encoded.txt: {}", total_tokens);
    Ok(())
}