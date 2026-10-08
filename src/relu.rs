//! Element-wise `max(0, x)`.


use crate::Layer;
use crate::Tensor;

pub struct Relu {
    cached_input: Option<Tensor>,
}

impl Relu {
    pub fn new() -> Self {
        Self { cached_input: None }
    }
}

impl Default for Relu {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Relu {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        self.cached_input = Some(input.clone());
        let mut output = Tensor::zeros(input.rows(), input.cols());
        for row in 0..input.rows() {
            for col in 0..input.cols() {
                let value = input[(row, col)];
                output[(row, col)] = if value > 0.0 { value } else { 0.0 };
            }
        }
        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let Some(input) = &self.cached_input else {
            panic!("relu backward requires forward first");
        };
        assert!(
            grad_output.rows() == input.rows() && grad_output.cols() == input.cols(),
            "relu gradient shape mismatch"
        );
        let mut grad_input = Tensor::zeros(grad_output.rows(), grad_output.cols());
        for row in 0..grad_output.rows() {
            for col in 0..grad_output.cols() {
                grad_input[(row, col)] = if input[(row, col)] > 0.0 {
                    grad_output[(row, col)]
                } else {
                    0.0
                };
            }
        }
        grad_input
    }
}
