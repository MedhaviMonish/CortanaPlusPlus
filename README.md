# Cortana++

Cortana++ is a lightweight machine learning engine built from scratch in C++ with CUDA acceleration.  
It is designed to explore the foundations of deep learning, custom GPU programming, and efficient model training without relying on heavy external libraries.

## Why Cortana++?

- To deeply understand and optimize the inner workings of machine learning at the system level.
- To build an independent, lightweight engine capable of running and training neural networks.
- To combine the power of C++ and CUDA for maximum performance and control.
- And I like HALO.

## Goals

- Implement core machine learning components (tensors, layers, optimizers) from scratch.
- Accelerate computation using custom CUDA kernels.
- Focus on simplicity, efficiency, and full transparency over the training process.
- Enable small-scale experiments in AI design, reasoning, and optimization.

---

> ⚙️ **Status:** Early development phase.  
> 🛠️ **Planned:** Dense layers, matrix operations, and basic backpropagation.

---

### 🚀 Completed: Custom CUDA Addition & Reduction Kernels

- Implemented elementwise tensor addition and scalar broadcasting using flat memory and memory-efficient chunking.
- Developed a fully parallel kernel to reduce arbitrary N-D tensors along the last axis.
- Handles large reductions in multiple stages using dynamic thread/block planning and staged accumulation.

---

### 🔢 Completed: Custom MatMul Kernel (Broadcasted Dot Product)

- Implemented a parallel matmul kernel that performs input × weight projection across rows.
- For each input vector, performs elementwise multiplication with every weight row and sums features to produce scalar outputs.
- Avoids explicit loops using thread-block mappings that simulate projection behavior without memory overhead.
- Serves as the foundation for Dense layer implementation and attention score computation.
- Implemented `DenseLayer` with full forward pass: `y = Wx + b`
- Added internal bias broadcasting (from `[1, M]` to `[N, M]`)
- Integrated optional activation handling: supports `Linear` and `ReLU`
- Created `Tensor::max`, `min`, and `clamp` methods for elementwise ops
- Implemented `ReLU` using `max(0)`
- Built and tested full end-to-end forward execution from input → matmul → bias → activation
- Added support for generating random tensors for testing and experimentation
---

### Row-major neuron layout — matches paper-style math and is easier to reason about.

Weights are defined with shape `[M, D]`:
- D = number of input features
- M = number of output neurons
- Each **row** represents one neuron's weights across all features.

In contrast, PyTorch/TensorFlow use `[D, M]`:
- Each **column** represents one neuron's weights (column-major layout)
- Requires input @ weight multiplication: `[N, D] @ [D, M] = [N, M]`
