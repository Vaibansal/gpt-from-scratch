# GPT from scratch

A character-level, decoder-only Transformer trained on Tiny Shakespeare, built from the "Let's build GPT" lecture.

| File | What it is |
| --- | --- |
| `bigram.py` | The baseline. Each character predicts the next from its own identity only. |
| `gpt.py` | The full model: multi-head causal self-attention, feed-forward layers, residual connections, LayerNorm and dropout. |

## Setup

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install torch
```

```bash
curl -L -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Run

Train the baseline:

```bash
.venv/bin/python bigram.py
```

Train the small Transformer:

```bash
.venv/bin/python gpt.py --preset small
```

Train the lecture's full model of about 10M parameters. It took 15 minutes on an A100 in the lecture and will take much longer on a laptop:

```bash
.venv/bin/python gpt.py --preset full
```

Continue an interrupted run from its last checkpoint, which is saved every 250 steps:

```bash
.venv/bin/python gpt.py --preset full --resume
```

Generate more text from a trained checkpoint:

```bash
.venv/bin/python gpt.py --preset small --generate 2000
```

## Fine-tune pretrained GPT-2

`finetune_gpt2.py` loads OpenAI's released GPT-2 weights (124M parameters, a 548 MB download) into a model written in the same style as `gpt.py`, then trains it briefly on Tiny Shakespeare. It needs three more packages:

```bash
.venv/bin/pip install tiktoken huggingface_hub safetensors
```

```bash
.venv/bin/python finetune_gpt2.py --prompt "ROMEO:"
```

```bash
.venv/bin/python finetune_gpt2.py --generate 500 --prompt "JULIET:"
```

The largest GPT-2 (1.5B parameters, a 6.4 GB download) doesn't fit in 24 GB of memory with every weight training, so train only its top layers:

```bash
.venv/bin/python finetune_gpt2.py --model gpt2-xl --train-layers 8 --batch-size 4 --max-iters 300 --learning-rate 3e-5
```

GPT-2 uses subword tokens of about 3.3 characters each, so the log also prints each validation loss divided by that, to compare with the character-level models.

Training saves a checkpoint to `gpt_<preset>.pt` at every evaluation and writes a sample to `sample_<preset>.txt` at the end. The scripts use CUDA or Apple's MPS GPU when available, and the CPU otherwise.

## Differences from the lecture

- Adds automatic MPS support, and the `small` preset so training finishes quickly on a laptop.
- Adds checkpoints and a `--generate` mode.
- The `full` preset matches the lecture's final hyperparameters, with a validation loss of about 1.48 reported there.

## Results

Training logs and generated samples are in [`results/`](results/). Model checkpoints are not committed (they are 4 MB to 5.8 GB).

| Model | Where | Validation loss |
| --- | --- | --- |
| Full character-level Transformer (~10M params), 5,000 steps | `results/train_full.log` | 1.499 (lecture reports about 1.48) |
| GPT-2 124M fine-tuned on Tiny Shakespeare | `results/finetune_gpt2.log` | 3.31 per token, about 1.00 per character |
| GPT-2 XL 1.5B, full fine-tune | `results/finetune_gpt2-xl.log` | ran out of memory on a 24 GB Apple M4, so train only the top layers |
