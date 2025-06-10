# Project DAO: A Model of Growing Intelligence

**DAO** is a research project dedicated to building a new kind of language model from first principles. Its philosophy is modeled on a developing mind: an intelligent system that can grow its own cognitive structure in response to confusion, learning not only new knowledge but new ways to comprehend that knowledge.

The model does not simply "think"; it **grows into understanding**.

## Core Concepts

The architecture is a synthesis of principles from neuroscience and modern machine learning, aiming to bridge the gap between interpretable sparse models and powerful dense models.

* **Sparse Basis Field:** The foundation of the model remains neuro-inspired. A `SpatialPooler` processes high-dimensional sensory inputs (like token and positional data) to form a **Basis Field** of sparse, distributed representations (SDRs). Each SDR represents a fundamental concept or feature learned from the data.

* **Resonant Density Representation (RDR):** This is the core of the new architecture. Instead of using sparse SDRs directly for sequence processing, we compose a dense, information-rich vector called an RDR.
    * A new **Resonance Layer** takes the active SDR from the Basis Field and, guided by context, learns to represent it as a **superposition** of many basis SDRs.
    * The resulting RDR is a dense, floating-point vector, analogous to the learned embeddings in a Transformer, but it is fundamentally **grounded** in the sparse concepts of the Basis Field. This approach seeks to combine the expressive power of dense representations with the semantic stability of sparse ones.

* **GPU-Accelerated Recurrent Memory:** The `TemporalMemory` has been evolved into a proper recurrent layer (similar to an RNN). It processes the dense RDRs to learn sequences and make predictions. By integrating the **LibTorch** library, these dense tensor operations are now GPU-accelerated, enabling modern, high-performance computation.

## The Learning Process

The learning process has been fundamentally redesigned to embrace the power of gradient-based optimization.

1.  **Tokenizer Training:** We continue to use Google's **SentencePiece** library to create a BPE (Byte-Pair Encoding) model from a text corpus, providing an optimized, fixed vocabulary.

2.  **Gradient-Based Assimilation:** The model's weights (in the `ResonanceLayer`, `TemporalMemory`, and output `vocab_matrix`) are now trained using backpropagation. The `trainer` uses the **Adam optimizer** to minimize a **Cross-Entropy loss** function, aligning the DAO's training with modern deep learning standards.

## How to Build and Use

The project requires CMake, a C++17 compiler, and several dependencies, most notably **LibTorch**.

### 1. Prerequisites
* A **C++17 compliant compiler** (e.g., GCC 9+, Clang 10+).
* **CMake** (version 3.10 or higher).
* **LibTorch (PyTorch C++ API):** Must be downloaded from the PyTorch website. The build is configured via the `Torch_DIR` variable in `CMakeLists.txt`.
* **SentencePiece Library**: Must be built and installed first.
* **(Optional) NVIDIA CUDA Toolkit:** For GPU acceleration.

### 2. Step-by-Step Workflow

1.  **Install Dependencies:** Install SentencePiece and download the LibTorch C++ distribution.
2.  **Configure CMake:** Edit `CMakeLists.txt` to set the `LIBTORCH_ROOT` variable to the correct path for your LibTorch installation.
3.  **Build the Project:** Run the build script (`./build.sh`). This creates the `train_tokenizer` and `chat` executables.
4.  **Train the Tokenizer:** Run `./train_tokenizer` on a text corpus to create the `tokenizer.model` file.
5.  **Run the Main Application:** Execute `./chat`.
    * **First Run:** If no `model.bin` is found, the application will initialize the model and begin the training and assimilation process on the data in the `./data` directory.
    * **Subsequent Runs:** If `model.bin` exists, it will be loaded, and you can begin interacting with the model immediately.

## The Path Forward

With the core architecture built and the training mechanism stabilized, the project moves from foundational engineering to the pursuit of greater intelligence.

1.  **Improve and Evaluate:** The immediate focus is on improving the quality of the current single-layer model. This involves more extensive training on larger datasets and rigorous evaluation on benchmark tasks to quantify its capabilities.
2.  **Evolve the Hierarchy:** The current model is a single layer. The next major architectural step is to extend the RDR and backpropagation concepts to a multi-layer hierarchy, allowing the model to learn progressively more abstract representations.
3.  **Explore Deeper Principles:** The ultimate goal remains to explore novel forms of machine intelligence. Future research will involve investigating hybrid learning models, differentiable spatial poolers, and more advanced representation theories.