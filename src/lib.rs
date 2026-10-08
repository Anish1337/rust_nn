//! A row-major matrix, linear layers, ReLU, and mean squared error.

mod layer;
mod linear;
mod loss;
mod network;
mod relu;
mod tensor;

pub use layer::Layer;
pub use linear::Linear;
pub use loss::MseLoss;
pub use network::NeuralNetwork;
pub use relu::Relu;
pub use tensor::Tensor;
