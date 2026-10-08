use rust_nn::{Linear, MseLoss, NeuralNetwork, Relu, Tensor};

fn main() -> std::process::ExitCode {
    let inputs = Tensor::from_rows(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
    let targets = Tensor::from_rows(&[[0.0], [1.0], [1.0], [0.0]]);

    let mut network = NeuralNetwork::new();
    network.add(Linear::new(2, 8, 42));
    network.add(Relu::new());
    network.add(Linear::new(8, 1, 43));

    let loss = MseLoss;
    let initial = loss.forward(&network.forward(&inputs), &targets);
    for _epoch in 0..10_000 {
        let prediction = network.forward(&inputs);
        let grad = loss.backward(&prediction, &targets);
        network.backward(&grad);
        network.update(0.05);
    }

    let prediction = network.forward(&inputs);
    let final_loss = loss.forward(&prediction, &targets);
    println!(
        "Initial MSE: {initial:.6e}\nFinal MSE: {final_loss:.6e}\nPredictions (00, 01, 10, 11):\n{prediction}"
    );

    if !final_loss.is_finite() || final_loss >= 0.001 || final_loss >= initial {
        return std::process::ExitCode::FAILURE;
    }
    for row in 0..targets.rows() {
        let value = prediction[(row, 0)];
        if !value.is_finite() || (value - targets[(row, 0)]).abs() >= 0.1 {
            return std::process::ExitCode::FAILURE;
        }
    }
    println!("XOR passed");
    std::process::ExitCode::SUCCESS
}
