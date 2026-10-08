//! Fully connected layer: `Y = XW + b`.


use std::mem;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::Layer;
use crate::Tensor;

enum Phase {
    Empty,
    Forwarded {
        input: Tensor,
    },
    Backward {
        grad_weights: Tensor,
        grad_bias: Tensor,
    },
}

pub struct Linear {
    weights: Tensor,
    bias: Tensor,
    phase: Phase,
}

impl Linear {
    /// He uniform weights in `[-sqrt(6 / inputs), +sqrt(6 / inputs)]`. Biases start at zero.
    pub fn new(inputs: usize, outputs: usize, seed: u64) -> Self {
        assert!(
            inputs > 0 && outputs > 0,
            "linear dimensions must be positive"
        );
        let mut weights = Tensor::zeros(inputs, outputs);
        let bound = (6.0 / inputs as f64).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);
        for row in 0..inputs {
            for col in 0..outputs {
                weights[(row, col)] = rng.random_range(-bound..bound);
            }
        }
        Self {
            weights,
            bias: Tensor::zeros(1, outputs),
            phase: Phase::Empty,
        }
    }

    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    pub fn grad_weights(&self) -> &Tensor {
        match &self.phase {
            Phase::Backward { grad_weights, .. } => grad_weights,
            _ => panic!("linear gradients are available after backward"),
        }
    }

    pub fn grad_bias(&self) -> &Tensor {
        match &self.phase {
            Phase::Backward { grad_bias, .. } => grad_bias,
            _ => panic!("linear gradients are available after backward"),
        }
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut output = input.matmul(&self.weights);
        output.add_row_vector(&self.bias);
        // Owned copy: the cache must stay valid after `input` is dropped.
        self.phase = Phase::Forwarded {
            input: input.clone(),
        };
        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // The block ends the borrow of `phase` before that field is replaced.
        let grad_weights;
        let grad_bias;
        let grad_input;
        {
            let Phase::Forwarded { input } = &self.phase else {
                panic!("linear backward requires forward first");
            };
            assert!(
                grad_output.rows() == input.rows() && grad_output.cols() == self.weights.cols(),
                "linear gradient shape mismatch"
            );
            grad_weights = input.transpose().matmul(grad_output);
            grad_bias = grad_output.sum_rows();
            grad_input = grad_output.matmul(&self.weights.transpose());
        }
        self.phase = Phase::Backward {
            grad_weights,
            grad_bias,
        };
        grad_input
    }

    fn update(&mut self, learning_rate: f64) {
        assert!(
            learning_rate.is_finite() && learning_rate >= 0.0,
            "learning rate must be finite and nonnegative"
        );
        assert!(
            matches!(self.phase, Phase::Backward { .. }),
            "linear update requires backward first"
        );
        // A field cannot be moved while `self` is borrowed. `replace` swaps
        // in `Empty` and hands back the gradients by value.
        let Phase::Backward {
            grad_weights,
            grad_bias,
        } = mem::replace(&mut self.phase, Phase::Empty)
        else {
            unreachable!("backward phase was confirmed");
        };
        let delta_weights = &grad_weights * learning_rate;
        let delta_bias = &grad_bias * learning_rate;
        self.weights -= &delta_weights;
        self.bias -= &delta_bias;
    }
}
