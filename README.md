# 🌙 EasyMuon

**Muon optimizer made easy!**

This is a single-file, plug-and-play implementation of the [Muon optimizer](https://github.com/KellerJordan/Muon/) designed for simplicity and stability. It combines **Muon** (for 2D hidden layers) and **AdamW** (for embeddings/biases) into a single, unified optimizer class.

> **What is Muon?** It uses orthogonal gradient updates via Newton-Schulz iterations to train Deep Models more efficiently than standard AdamW.

## ✨ Features

*   🚀 **Drop-in Replacement**: Just copy `easy_muon.py` into your project.
*   🛡️ **Automatic Fallback**: Applies Muon to 2D matrices and AdamW to everything else automatically.
*   ⚡ **Optimized**: Uses many fused operations in `torch`.
*   🤝 **Distributed Ready**: Works out-of-the-box with DDP and FSDP.
*   🌚 **Moonlight Scaling**: Implements the recommended scaling from [Moonlight](https://github.com/MoonshotAI/Moonlight), allowing you to reuse standard AdamW hyperparameters.

## 📦 Usage

### 1. Basic Setup (Recommended)

Just grab the file and let the helper function do the work:

```python
from easy_muon import Muon, build_muon_param_groups

# 1. Automatically split params into Muon (matrices) and AdamW (biases/embeds)
param_groups = build_muon_param_groups(model)

# 2. Initialize optimizer
optimizer = Muon(
    param_groups,
    lr=3e-4,
    weight_decay=0.1,
    momentum=0.95
)

# 3. Train as usual!
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 2. Advanced / Manual Configuration

You can fully customize which parameters go where:

```python
optimizer = Muon([
    # Group 1: The heavy lifters (Muon)
    {
        'params': model.layers.parameters(), 
        'use_muon': True, 
        'lr': 3e-4, 
        'ns_steps': 5
    },
    # Group 2: The sensitive parts (AdamW)
    {
        'params': model.embeddings.parameters(), 
        'use_muon': False, 
        'lr': 1e-4, 
        'adamw_betas': (0.9, 0.999)
    }
])
```

## ⚙️ Scaling Modes

This implementation supports two scaling strategies for the Muon update:

1.  **`"moonlight"` (Default & Recommended):**
    *   Balances updates to match AdamW's RMS.
    *   **Benefit:** You don't need to tune learning rates from scratch; standard AdamW LRs usually work immediately.
2.  **`"original"`:**
    *   The original scaling from the Muon paper/repo.
    *   **Note:** Requires significantly higher learning rates (e.g., ~0.02) to be effective.

## 📜 References

*   [**Muon (Keller Jordan)**](https://github.com/KellerJordan/Muon/): The original implementation.
*   [**Moonlight**](https://github.com/MoonshotAI/Moonlight): Source of the improved scaling logic.
