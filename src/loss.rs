//! Mean squared error, kept outside the layer list.


use crate::Tensor;

pub struct MseLoss;

impl MseLoss {
    pub fn forward(&self, prediction: &Tensor, target: &Tensor) -> f64 {
        validate(prediction, target);
        let mut total = 0.0;
        for row in 0..prediction.rows() {
            for col in 0..prediction.cols() {
                let error = prediction[(row, col)] - target[(row, col)];
                total += error * error;
            }
        }
        total / prediction.len() as f64
    }

    #[must_use]
    pub fn backward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        validate(prediction, target);
        let scale = 2.0 / prediction.len() as f64;
        &(prediction - target) * scale
    }
}

fn validate(prediction: &Tensor, target: &Tensor) {
    assert!(
        prediction.rows() == target.rows()
            && prediction.cols() == target.cols()
            && !prediction.is_empty(),
        "mse requires matching, nonempty tensors"
    );
}
