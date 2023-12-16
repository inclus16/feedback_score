use std::sync::Arc;

use burn::data::dataloader::batcher::Batcher;
use burn_tensor::{Data, ElementConversion, Int, Tensor};
use burn_tensor::backend::{AutodiffBackend, Backend};

use crate::data::batch::CsvBatch;
use crate::data::dataset::DatasetItem;
use crate::preprocessing::tfidf::TfIdfVectorizer;

pub struct TrainBatcher<B: AutodiffBackend> {
    device: B::Device,
    tokenizer: Arc<TfIdfVectorizer>,
}

impl<B: AutodiffBackend> TrainBatcher<B> {
    pub fn new(device: B::Device, tokenizer: Arc<TfIdfVectorizer>) -> Self {
        Self { device, tokenizer }
    }

    fn classes_from_label(&self, label: u8) -> Tensor<B, 1, Int>
    {
        let mut classes = [0; 5];
        classes[label as usize - 1] = 1;
        Tensor::from_ints(Data::from(classes))
    }
}


impl<B: AutodiffBackend> Batcher<DatasetItem, CsvBatch<B>> for TrainBatcher<B>
{
    fn batch(&self, items: Vec<DatasetItem>) -> CsvBatch<B> {
        let mut text = Vec::with_capacity(items.len());
        let mut targets = Vec::with_capacity(items.len());
        for d in items {
            let tokenized = self.tokenizer.process(d.text);
            text.push(Tensor::from_floats(Data::from(tokenized.as_slice())).reshape([1, tokenized.len()]));
            targets.push(Tensor::from_ints(Data::from([(d.label as i32 - 1).elem()])));
        }
        return CsvBatch {
            text: Tensor::cat(text, 0).to_device(&self.device),
            targets: Tensor::cat(targets, 0).to_device(&self.device),
        };
    }
}


pub struct ValBatcher<B: Backend> {
    device: B::Device,
    tokenizer: Arc<TfIdfVectorizer>,
}

impl<B: Backend> ValBatcher<B> {
    pub fn new(device: B::Device, tokenizer: Arc<TfIdfVectorizer>) -> Self {
        Self { device, tokenizer }
    }

    fn classes_from_label(&self, label: u8) -> Tensor<B, 1, Int>
    {
        let mut classes = [0; 5];
        classes[label as usize - 1] = 1;
        Tensor::from_ints(Data::from(classes))
    }
}


impl<B: Backend> Batcher<DatasetItem, CsvBatch<B>> for ValBatcher<B>
{
    fn batch(&self, items: Vec<DatasetItem>) -> CsvBatch<B> {
        let mut text = Vec::with_capacity(items.len());
        let mut targets = Vec::with_capacity(items.len());
        for d in items {
            let tokenized = self.tokenizer.process(d.text);
            text.push(Tensor::from_floats(Data::from(tokenized.as_slice())).reshape([1, tokenized.len()]));
            targets.push(Tensor::from_ints(Data::from([(d.label as i32 - 1).elem()])));
        }
        return CsvBatch {
            text: Tensor::cat(text, 0).to_device(&self.device),
            targets: Tensor::cat(targets, 0).to_device(&self.device),
        };
    }
}



