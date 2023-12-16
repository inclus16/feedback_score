use burn_tensor::{Float, Int, Tensor};
use burn_tensor::backend::Backend;

#[derive(Clone, Debug)]
pub struct CsvBatch<B: Backend>
{
    pub text: Tensor<B, 2, Float>,
    pub targets: Tensor<B, 1, Int>,
}