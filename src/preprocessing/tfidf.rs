use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::preprocessing::tokenizer::Tokenizer;

#[derive(Deserialize, Serialize)]
struct SavedData
{
    pub tokens_count_in_all_documents: Vec<TokenInAllDocumentsCount>,
    pub documents_count: usize,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
struct TokenInAllDocumentsCount
{
    token: String,
    count: usize,
}

pub struct TfIdfVectorizer
{
    tokenizer: Tokenizer,
    token_indexes: HashMap<String, usize>,
    tokens_count_in_all_documents: Vec<TokenInAllDocumentsCount>,
    documents_count: usize,
}

impl TfIdfVectorizer
{
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer, tokens_count_in_all_documents: vec![], documents_count: 0, token_indexes: HashMap::new() }
    }

    pub fn get_tokens_len(&self) -> usize
    {
        return self.tokens_count_in_all_documents.len();
    }

    pub fn cut(&mut self, min_count: usize)
    {
        println!("Cutting idf to min count: {}", min_count);
        println!("Before cut: {:?}", self.tokens_count_in_all_documents.len());
        let min_count_min_pos = self.tokens_count_in_all_documents.binary_search_by(|x| x.count.cmp(&min_count));
        match min_count_min_pos {
            Ok(p) => {
                if p < 2 {
                    println!("No documents asserts min count! Decrease min_count in cut, or increase your dataset size");
                }
                self.tokens_count_in_all_documents.drain(0..p - 1);
            }
            Err(_) => {
                println!("No documents asserts min count! Decrease min_count in cut, or increase your dataset size");
            }
        }
        println!("After cut: {:?}", self.tokens_count_in_all_documents.len());
    }

    pub fn sort(&mut self)
    {
        self.tokens_count_in_all_documents.sort_by(|a, b| a.count.cmp(&b.count));
    }

    pub fn save(&self, path: &str)
    {
        let mut file: File;
        if Path::new(&path).is_file() {
            file = File::open(path).unwrap();
        } else {
            file = File::create(path).unwrap();
        }
        file.write(&rmp_serde::to_vec(&SavedData {
            tokens_count_in_all_documents: self.tokens_count_in_all_documents.clone(),
            documents_count: self.documents_count,
        }).unwrap()).unwrap();
    }

    pub fn load(&mut self, path: &str)
    {
        let mut file = File::open(path).unwrap();
        let mut buff: Vec<u8> = Vec::new();
        file.read_to_end(&mut buff).unwrap();
        let data = rmp_serde::from_slice::<SavedData>(&buff).unwrap();
        self.tokens_count_in_all_documents = data.tokens_count_in_all_documents;
        self.documents_count = data.documents_count;
        println!("Tokens count :{:?}", self.tokens_count_in_all_documents.len());
    }
    pub fn remember_one(&mut self, text: String)
    {
        self.documents_count += 1;
        let tokens = self.tokenizer.encode(text);
        for token in tokens.iter() {
            let index = self.token_indexes.get(token);
            match index {
                None => {
                    let token_count = TokenInAllDocumentsCount {
                        token: token.clone(),
                        count: 1,
                    };
                    self.tokens_count_in_all_documents.push(token_count);
                    self.token_indexes.insert(token.clone(), self.tokens_count_in_all_documents.len() - 1);
                }
                Some(i) => {
                    let token_count: &mut TokenInAllDocumentsCount = self.tokens_count_in_all_documents.get_mut(*i).unwrap();
                    token_count.count += 1;
                }
            }
        }
    }

    pub fn process(&self, data: String) -> Vec<f32>
    {
        let tokens = self.tokenizer.encode(data);
        let mut map_count: HashMap<String, f32> = HashMap::with_capacity(tokens.len());
        for token in tokens.iter() {
            match map_count.get_mut(token) {
                None => {
                    map_count.insert(token.clone(), 1_f32);
                }
                Some(c) => {
                    *c += 1_f32;
                }
            };
        };
        return self.tokens_count_in_all_documents.iter().map(|dc| {
            match map_count.get(&dc.token) {
                None => 0_f32,
                Some(count_in_document) => {
                    return count_in_document * (self.documents_count as f32 / dc.count as f32).log2();
                }
            }
        }).collect::<Vec<f32>>();
    }
}