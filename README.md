# Memory Layers with CPU Pinned Memory

This project implements **Memory Layers** utilizing **CPU Pinned Memory**, based on the paper [Memory Layers at Scale](https://ai.meta.com/research/publications/memory-layers-at-scale/).

By storing large-scale embeddings in CPU memory and fetching only the necessary parts to the GPU during computation, this implementation significantly minimizes GPU memory usage while maintaining training efficiency.

## Key Features

  - **CPU Pinned Memory**: Efficient CPU-GPU data transfer using PyTorch's pinned memory.
  - **Unique Indices Optimization**: Minimizes data transfer bandwidth by removing duplicate indices before fetching.
  - **Large-Scale Memory Layer**: scalable architecture based on Product Key Memory (PKM).
  - **Optimized Training**: Supports FP32 precision training with optimized memory management.

## Project Structure

```
📦memory
 ┣ 📂lingua                    # Core Library
 ┃ ┣ 📂product_key            # Memory Layer Implementation
 ┃ ┃ ┣ 📜memory.py            # Main Logic for Product Key Memory
 ┃ ┃ ┣ 📜zero_copy.py         # CPU Pinned Memory Implementation
 ┃ ┃ ┗ 📜colwise_embeddingbag.py
 ┃ ┗ 📜transformer.py
 ┣ 📂apps/main                # Main Application
 ┃ ┣ 📜train.py               # Training Script
 ┃ ┗ 📂configs                # Configuration Files
 ┣ 📂setup                    # Setup Scripts
 ┣ 📂agi_test                 # Experiment Configs & Scripts
 ┗ 📂tokenizer                # Tokenizer Files
```

## Quick Start

### 1\. Environment Setup

```bash
# Clone the repository
git clone https://github.com/hxeek96/agi_test_cpl.git
cd memory

# Create Conda environment and install packages
bash setup/create_env.sh

# Activate environment
conda activate prj_hs
```

**Prerequisites:**

  - Ubuntu 20.04 or higher

### 2\. Data Preparation

```bash
# Download and prepare Hugging Face dataset
python setup/download_prepare_hf_data.py fineweb_edu_10bt <MEMORY_GB> \
    --data_dir ./data \
    --seed 42

# Download Tokenizer (Llama3)
python setup/download_tokenizer.py llama3 ./tokenizer \
    --api_key <HUGGINGFACE_TOKEN>
```

## Experiments

### Test 1: Training with Zero-Copy

Run the training script using the Zero-Copy memory layer.

```bash
bash agi_test/test_1/script/zero_copy_train.sh
```

**Configuration:** [agi\_test/test\_1/config/zero\_copy.yaml](https://www.google.com/search?q=agi_test/test_1/config/zero_copy.yaml)

**Key Settings:**

```yaml
model:
  productkey_args:
    zero_copy: true           # Enable CPU Pinned Memory
    mem_offload: false        # Disable FPGA offload
    mem_n_keys: 8000          # Number of memory keys
    mem_share_values: true    # Enable value sharing
    mem_knn: 32               # K-nearest neighbors
    mem_k_dim: 512            # Key dimension
```

### Test 2: Evaluation

Run the evaluation on tasks such as HellaSwag, PIQA, and NQ Open.

```bash
bash agi_test/test_2/script/eval_zerop_copy.sh
```

### ⚠️ Critical Path Configuration

Before running the scripts, ensure the following paths in your config files match your local environment:

1.  `data.root_dir`: Path to the dataset directory.
2.  `data.tokenizer.path`: Path to the tokenizer directory.
3.  `dump_dir`: Checkpoint saving path (specified in the shell script).

## Core Implementation

### CPU Pinned Memory Layer

This layer stores memory values in CPU pinned memory instead of GPU memory. It fetches only the required embeddings during the forward pass.

**Optimization Techniques:**

1.  **Unique Indices**: Removes duplicate indices to reduce the volume of data transferred between CPU and GPU.
2.  **Pinned Memory**: Uses pinned (page-locked) memory for faster host-to-device transfer.
3.  **BFloat16 Compression**: Compresses data to BFloat16 during transfer to save bandwidth.
4.  **Non-blocking Transfer**: Utilizes asynchronous transfer to minimize wait times.

<!-- end list -->

```python
# lingua/product_key/zero_copy.py
class ZeroCopy(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        # Store values in CPU pinned memory
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device="cpu").pin_memory()
        )

    def forward(self, indices, scores):
        # 1. Identify unique indices on GPU to minimize transfer size
        unique_indices, inverse = torch.unique(indices.flatten(), return_inverse=True)
        
        # 2. Lookup embeddings from CPU Pinned Memory
        unique_emb_cpu = self.weight[unique_indices.cpu()]
        
        # 3. Compress to BFloat16 and transfer to GPU
        unique_emb_gpu = unique_emb_cpu.to(dtype=torch.bfloat16, device=indices.device)
        
        # 4. Restore original shape and apply weighted sum
        return weighted_sum(unique_emb_gpu[inverse], scores)
```

## Hyperparameters

| Parameter | Default | Description |
|:---|:---|:---|
| `mem_n_keys` | 8000 | Number of memory keys |
| `mem_knn` | 32 | Number of nearest neighbors to retrieve |
| `mem_k_dim` | 512 | Dimension of the key embeddings |
| `mem_heads` | 2 | Number of memory heads |
| `batch_size` | 2 | Batch size per device |
| `seq_len` | 4096 | Sequence length |

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors, try the following:

  - Decrease `batch_size`.
  - Decrease `mem_n_keys`.
  - Set the environment variable: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## References & Acknowledgments

This code is based on Meta's [Memory Layer](https://www.google.com/search?q=https://github.com/facebookresearch/memory/tree/main) repository, with additional implementations for zero-copy functionality.