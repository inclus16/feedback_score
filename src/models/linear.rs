use burn::{
    nn::{
        Linear, LinearConfig,
    },
    tensor::{backend::Backend, Tensor},
};
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig};
use burn::nn::loss::CrossEntropyLoss;
use burn::tensor::Float;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};
use burn_tensor::activation::{sigmoid, softmax, tanh};
use burn_tensor::backend::AutodiffBackend;
use burn_tensor::Int;

use crate::data::batch::CsvBatch;

#[derive(Config)]
pub struct LinearModelConfig
{
    input_size: usize,
}

#[derive(Module, Debug)]
pub struct LinearModel<B: Backend> {
    linear1: Linear<B>,
    dropout1: Dropout,
    linear2: Linear<B>,
    dropout2: Dropout,
    linear3: Linear<B>,
}

impl LinearModelConfig
{
    pub fn init<B: Backend>(&self) -> LinearModel<B> {
        LinearModel {
            linear1: LinearConfig::new(self.input_size, 256).init(),
            dropout1: DropoutConfig::new(0.2).init(),
            linear2: LinearConfig::new(256, 128).init(),
            dropout2: DropoutConfig::new(0.2).init(),
            linear3: LinearConfig::new(128, 5).init(),
        }
    }

    pub fn init_with<B: Backend>(&self, record: LinearModelRecord<B>) -> LinearModel<B>
    {
        LinearModel {
            linear1: LinearConfig::new(self.input_size, 256).init_with(record.linear1),
            dropout1: DropoutConfig::new(0.2).init(),
            linear2: LinearConfig::new(256, 128).init_with(record.linear2),
            dropout2: DropoutConfig::new(0.2).init(),
            linear3: LinearConfig::new(128, 5).init_with(record.linear3),
        }
    }
}

impl<B: Backend> LinearModel<B> {
    pub fn infer(&self, item: Tensor<B, 2, Float>) -> Tensor<B, 2, Float>
    {
        let x = item;
        let x = self.linear1.forward(x);
        let x = tanh(x);
        let x = self.linear2.forward(x);
        let x = tanh(x);
        let x = self.linear3.forward(x);
        let x = sigmoid(x);
        softmax(x, 1)
    }
    pub fn forward_classification(
        &self,
        input: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let x = input;
        let x = self.linear1.forward(x);
        let x = tanh(x);
        let x = self.dropout1.forward(x);
        let x = self.linear2.forward(x);
        let x = tanh(x);
        let x = self.dropout2.forward(x);
        let x = self.linear3.forward(x);
        let output = sigmoid(x);
        let loss = CrossEntropyLoss::default();
        let loss = loss.forward(output.clone(), targets.clone());
        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<CsvBatch<B>, ClassificationOutput<B>> for LinearModel<B> {
    fn step(&self, batch: CsvBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.text, batch.targets);
        let grads = item.loss.backward();
        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<CsvBatch<B>, ClassificationOutput<B>> for LinearModel<B> {
    fn step(&self, batch: CsvBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.text, batch.targets)
    }
}