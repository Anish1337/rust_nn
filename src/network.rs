//! Ordered chain of layers.


use crate::Layer;
use crate::Tensor;

pub struct NeuralNetwork {
    layers: Vec<Box<dyn Layer>>,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add<T>(&mut self, layer: T)
    where
        T: Layer + 'static,
    {
        self.layers.push(Box::new(layer));
    }

    /// An empty network returns a clone of `input`.
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    /// Walks layers from the output back toward the input.
    pub fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let mut gradient = grad_output.clone();
        for layer in self.layers.iter_mut().rev() {
            gradient = layer.backward(&gradient);
        }
        gradient
    }

    pub fn update(&mut self, learning_rate: f64) {
        for layer in &mut self.layers {
            layer.update(learning_rate);
        }
    }
}

impl Default for NeuralNetwork {
    fn default() -> Self {
        Self::new()
    }
}
