# 📝 Task Tracker

## 🔮 Future
- [ ] Explore hybrid learning models (gradient-based training + online Hebbian adaptation).
- [ ] Investigate making the Spatial Pooler a differentiable module.
- [ ] Research complex-valued or holographic representations for advanced binding.
- [ ] Extend the RDR architecture to a multi-layer hierarchy.

## 📥 Backlog
- [ ] **Train on a significantly larger dataset.**
- [ ] **Increase the number of training epochs and tune hyperparameters (e.g., learning rate).**
- [ ] Rigorously evaluate the trained model on benchmark datasets.
- [ ] Analyze and improve generation quality through architectural or training enhancements.

## ⏳ Pending / In-Progress
- [ ] _(cleared)_

## ✅ Done
- [x] **Successfully run the training and generation process on the GPU.**
- [x] Resolve all system, environment, and library-linking issues.
- [x] Implement the backpropagation learning algorithm in `trainer.cpp` using LibTorch's autograd engine.
- [x] Convert the `vocab_matrix` from Eigen to a `torch::Tensor` throughout the project.
- [x] Integrate the LibTorch library for GPU-accelerated computation.
- [x] Create the new `ResonanceLayer` class for composing dense RDRs.
- [x] Refactor the `TemporalMemory` into an RNN-like layer that processes `torch::Tensor` objects.
- [x] Refactor the `ConversationalGenerator` to use the new tensor-based architecture.
- [x] Create a new `dao_model.cpp` with `torch`-based model serialization.
- [x] Stabilize the training loop and resolve all runtime `NaN` errors.