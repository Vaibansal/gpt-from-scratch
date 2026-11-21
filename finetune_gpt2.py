"""Fine-tune OpenAI's pretrained GPT-2 on Tiny Shakespeare.

Train:     python finetune_gpt2.py
Generate:  python finetune_gpt2.py --generate 1000 --prompt "ROMEO:"

Same architecture as gpt.py, with the details GPT-2 adds: a subword (BPE)
tokenizer, all heads computed in one batched operation, GELU, and the output
layer sharing weights with the token embedding.
"""
import argparse
import time
from pathlib import Path

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

CONFIGS = {
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M parameters
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M parameters
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M parameters
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M parameters
}
VOCAB_SIZE = 50257
N_POSITIONS = 1024  # longest context GPT-2 was trained with

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--model", choices=CONFIGS, default="gpt2")
parser.add_argument("--max-iters", type=int, default=500)
parser.add_argument("--learning-rate", type=float, default=6e-5)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--block-size", type=int, default=256, help="training context, in tokens")
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--train-layers", type=int, metavar="N",
                    help="train only the top N blocks and freeze the rest, to fit in memory")
parser.add_argument("--eval-interval", type=int, default=50)
parser.add_argument("--eval-iters", type=int, default=40)
parser.add_argument("--generate", type=int, metavar="N",
                    help="skip training and print N tokens from the saved checkpoint")
parser.add_argument("--prompt", default="\n", help="text to start generating from")
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=200)
parser.add_argument("--device", help="cuda, mps or cpu (default: the fastest available)")
args = parser.parse_args()

if args.device:
    device = args.device
elif torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(1337)

root = Path(__file__).parent
checkpoint_path = root / f"{args.model}_shakespeare.pt"
sample_path = root / f"sample_{args.model}.txt"
enc = tiktoken.get_encoding("gpt2")


class CausalSelfAttention(nn.Module):
    """All heads at once: heads become a batch dimension, as in nanoGPT."""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.dropout = dropout
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # query, key and value in one matmul
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # the same masked, scaled softmax(q k^T) v as gpt.py's Head, in one fused kernel
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # concatenate the heads
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU(approximate="tanh")  # the GELU variant GPT-2 used
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, n_embd)
        self.wpe = nn.Embedding(N_POSITIONS, n_embd)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList(Block(n_embd, n_head, dropout) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -N_POSITIONS:])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                # only sample from the k most likely tokens
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


def load_pretrained(model, name):
    """Copy OpenAI's released weights from Hugging Face into our module."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download(repo_id=f"openai-community/{name}", filename="model.safetensors")
    theirs = load_file(path)
    ours = model.state_dict()
    copied = set()
    with torch.no_grad():
        for key, value in theirs.items():
            key = key.removeprefix("transformer.")
            if key.endswith((".attn.bias", ".attn.masked_bias")):
                continue  # their causal mask buffers; we use is_causal instead
            # OpenAI stored these layers as Conv1D, which is a transposed Linear
            if key.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")):
                value = value.t()
            assert ours[key].shape == value.shape, (key, ours[key].shape, value.shape)
            ours[key].copy_(value)
            copied.add(key)
    missing = set(ours) - copied - {"lm_head.weight"}  # lm_head is tied to wte
    assert not missing, f"weights not found in the checkpoint: {sorted(missing)}"


def main():
    model = GPT2(dropout=args.dropout, **CONFIGS[args.model]).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"{args.model} on {device}: {params / 1e6:.1f}M parameters", flush=True)

    if args.generate is not None:
        if not checkpoint_path.exists():
            raise SystemExit(f"No checkpoint at {checkpoint_path}. Fine-tune first.")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(sample(model, args.prompt, args.generate))
        return

    load_pretrained(model, args.model)

    text = (root / "input.txt").read_text(encoding="utf-8")
    data = torch.tensor(enc.encode_ordinary(text), dtype=torch.long)
    # gpt.py's losses are per character; these are per token, which covers several
    chars_per_token = len(text) / len(data)
    print(f"{len(data):,} tokens, {chars_per_token:.2f} characters per token", flush=True)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}

    def get_batch(split):
        source = splits[split]
        ix = torch.randint(len(source) - args.block_size, (args.batch_size,))
        x = torch.stack([source[i : i + args.block_size] for i in ix])
        y = torch.stack([source[i + 1 : i + args.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                _, loss = model(*get_batch(split))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    if args.train_layers is not None:
        # frozen weights need no gradients or optimizer state, which is most of the memory
        for p in model.parameters():
            p.requires_grad = False
        for module in [*model.h[-args.train_layers:], model.ln_f]:
            for p in module.parameters():
                p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"training {sum(p.numel() for p in trainable) / 1e6:.1f}M parameters", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    best_val = float("inf")
    start = time.time()
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss()
            improved = losses["val"] < best_val
            print(
                f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} "
                f"(per character {losses['val'] / chars_per_token:.4f}) "
                f"({(time.time() - start) / 60:.1f} min)" + (" best" if improved else ""),
                flush=True,
            )
            if improved:
                best_val = losses["val"]
                torch.save(model.state_dict(), checkpoint_path)
        if it == args.max_iters:
            break
        _, loss = model(*get_batch("train"))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    text_out = sample(model, args.prompt, 500)
    sample_path.write_text(text_out, encoding="utf-8")
    print(text_out)
    print(f"\nSaved {checkpoint_path.name} and {sample_path.name}")


def sample(model, prompt, num_tokens):
    model.eval()
    idx = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, num_tokens, temperature=args.temperature, top_k=args.top_k)
    return enc.decode(out[0].tolist())


if __name__ == "__main__":
    main()
