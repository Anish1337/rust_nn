# rust_nn

A rust rewrite of cpp_ml, a small feed-forward network that learns XOR. 

## Run

```bash
cargo run --example xor
```

The example trains for 10,000 epochs at learning rate 0.05. It prints `XOR passed` when the final mean squared error is below 0.001 and every prediction is within 0.1 of its target.

## Rust features for learning

| File | Feature |
| --- | --- |
| `src/tensor.rs` | owned `Vec`, explicit `Clone`, `Index`, `Add`/`Mul`, `Display` |
| `src/layer.rs` | trait, default method, object safety for `dyn` |
| `src/network.rs` | `Box<dyn Layer>`, `'static`, `where`, reverse iteration |
| `src/linear.rs` | state `enum`, seeded `StdRng`, `mem::replace` |
| `src/relu.rs` | `Option` cache, `let`/`else` |
| `src/loss.rs` | unit struct, panic on a bad call |
| `examples/xor.rs` | crate API, `ExitCode` |

