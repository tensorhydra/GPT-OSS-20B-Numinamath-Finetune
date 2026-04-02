# LoRA Fine-Tuning for MoE LLMs on H100

> Memory-optimized LoRA fine-tuning for Mixture-of-Experts large language models — tested on a 20B MoE model (`gpt-oss-20b`) with the NuminaMath-TIR dataset on a single H100 GPU.

---

## Overview

This notebook implements full-precision LoRA fine-tuning specifically engineered for **Mixture-of-Experts (MoE)** architecture models. Standard fine-tuning scripts often break on MoE models due to CPU offloading conflicts with expert routing layers. This solution targets only attention layers with LoRA adapters, sidesteps those pitfalls, and squeezes training onto a single H100 with aggressive memory optimizations.

Link to model on HF: https://huggingface.co/tensorhydra/gpt-oss-20b-numinamath

**Key features:**
- MoE-compatible LoRA configuration (attention-only targeting)
- Flash Attention 2 integration
- BF16 precision + TF32 + fused AdamW optimizer
- Gradient checkpointing (non-reentrant mode)
- Real-time GPU memory monitoring
- Automatic train/validation splitting from CSV
- Flexible column name detection for prompt/response fields

---

## Model & Dataset

| Component | Details |
|---|---|
| **Base Model** | `danielhanchen/gpt-oss-20b` (Mixture of Experts, 20B params) |
| **Dataset** | [NuminaMath-TIR](https://www.kaggle.com/datasets/jorgeplazas/numinamath-tir) — math reasoning with tool-integrated reasoning |
| **Task** | Supervised instruction fine-tuning (causal LM) |
| **Format** | ChatML (`<\|im_start\|>` / `<\|im_end\|>`) |

---

## Requirements

```bash
pip install flash-attn --no-build-isolation
pip install transformers peft datasets torch pandas
```

**Hardware:** Single NVIDIA H100 (80GB recommended for 20B MoE)

**Software:**
- Python 3.10+
- PyTorch 2.x
- Transformers 4.40+
- PEFT 0.10+
- Flash Attention 2

---

## Configuration

All hyperparameters live in the `Config` dataclass. Key settings:

```python
config = Config(
    # Model
    model_name="danielhanchen/gpt-oss-20b",

    # LoRA — attention layers only (MoE-safe)
    lora_r=64,
    lora_alpha=128,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],

    # Training
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # effective batch = 16
    learning_rate=2e-4,
    max_seq_length=8192,

    # Precision
    bf16=True,
    tf32=True,
    optim="adamw_torch_fused",
)
```

### LoRA Settings Explained

| Parameter | Value | Reason |
|---|---|---|
| `lora_r` | 64 | Higher rank for complex math reasoning |
| `lora_alpha` | 128 | 2× rank — standard scaling |
| `target_modules` | attention only | MoE expert layers cause KeyError with LoRA |
| `lora_dropout` | 0.05 | Light regularization |
| `use_cpu_offload` | `False` | **Critical** — breaks MoE expert routing |

---

## Why MoE Models Need Special Handling

Standard LoRA fine-tuning scripts fail on MoE architectures because:

1. **CPU offloading** conflicts with expert dispatch — when experts are offloaded to CPU, the routing mechanism throws `KeyError` during forward passes
2. **MLP/expert layer LoRA** can destabilize learned routing distributions
3. **`device_map` with max_memory** can split expert groups across devices incorrectly

This notebook addresses all three by disabling offloading, targeting only attention projections, and using `device_map="auto"` to let the model handle its own MoE layout.

---

## Data Format

The dataset loader auto-detects common column names:

| Column Type | Accepted Names |
|---|---|
| Prompt/Input | `problem`, `question`, `input`, `prompt` |
| Response/Output | `solution`, `answer`, `output`, `response` |

The CSV is split into train/validation sets (default: 95% / 5%) before tokenization.

Each example is formatted as:

```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
```

---

## Training Output

The trainer logs to stdout with step-by-step metrics:

```
================================================================================
Step: 250/3800
Epoch: 0.13/2.00
Training Loss: 1.2847
Learning Rate: 1.98e-04

[Memory] Step 250: Allocated: 52.3GB, Reserved: 58.1GB, Peak: 54.7GB
================================================================================
EVALUATION - Step: 250
Validation Loss: 1.3102
Change from previous: +0.0000
================================================================================
```

Checkpoints and final LoRA adapters are saved to `./lora_outputs/`.

---

## Loading the Fine-Tuned Model

The notebook saves only the LoRA adapter weights (not the full model). To run inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "danielhanchen/gpt-oss-20b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./lora_outputs/final_model")
tokenizer = AutoTokenizer.from_pretrained("./lora_outputs/final_model")
```

---

## Memory Optimization Tips

If you hit OOM errors, try these in order:

1. Reduce `max_seq_length`: `8192 → 4096 → 2048`
2. Reduce `lora_r`: `64 → 32 → 16`
3. Drop to `["q_proj", "v_proj"]` only for LoRA targets
4. Switch to QLoRA (4-bit quantization) — add `load_in_4bit=True` to `from_pretrained`

---

## Project Structure

```
.
├── notebook.ipynb          # Main training notebook
├── lora_outputs/           # Saved checkpoints and final adapters
│   ├── checkpoint-250/
│   ├── checkpoint-500/
│   └── final_model/
└── README.md
```

---

## License

This project is released under the MIT License.
