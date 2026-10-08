//! Shared layer contract.

use crate::Tensor;

pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> Tensor;
    fn backward(&mut self, grad_output: &Tensor) -> Tensor;
    /// Activations have no parameters
    fn update(&mut self, _learning_rate: f64) {}
}
