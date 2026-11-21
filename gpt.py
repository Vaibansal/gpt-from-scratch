"""A decoder-only Transformer trained on Tiny Shakespeare, one character at a time.

Train:     python gpt.py --preset small
Generate:  python gpt.py --preset small --generate 2000
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

PRESETS = {
    # sized to finish in minutes on a laptop
    "small": dict(
        batch_size=32, block_size=128, max_iters=3000, learning_rate=1e-3,
        n_embd=128, n_head=4, n_layer=4, dropout=0.1,
    ),
    # the lecture's final model: about 10M parameters, 15 minutes on an A100
    "full": dict(
        batch_size=64, block_size=256, max_iters=5000, learning_rate=3e-4,
        n_embd=384, n_head=6, n_layer=6, dropout=0.2,
    ),
    # wider and deeper than the lecture, with more dropout because the full model overfits
    "large": dict(
        batch_size=64, block_size=256, max_iters=5000, learning_rate=3e-4,
        n_embd=512, n_head=8, n_layer=8, dropout=0.3,
    ),
}

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--preset", choices=PRESETS, default="small")
parser.add_argument("--max-iters", type=int, help="override the preset's training steps")
parser.add_argument("--generate", type=int, metavar="N",
                    help="skip training and print N characters from the saved checkpoint")
parser.add_argument("--resume", action="store_true",
                    help="continue training from the last saved checkpoint")
parser.add_argument("--device", help="cuda, mps or cpu (default: the fastest available)")
args = parser.parse_args()

config = PRESETS[args.preset]
batch_size = config["batch_size"]
block_size = config["block_size"]
learning_rate = config["learning_rate"]
n_embd = config["n_embd"]
n_head = config["n_head"]
n_layer = config["n_layer"]
dropout = config["dropout"]
max_iters = args.max_iters if args.max_iters is not None else config["max_iters"]
eval_interval = 250
eval_iters = 100

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
data_path = root / "input.txt"
checkpoint_path = root / f"gpt_{args.preset}.pt"
state_path = root / f"gpt_{args.preset}.state.pt"  # optimizer and step, for --resume
sample_path = root / f"sample_{args.preset}.txt"

if not data_path.exists():
    raise SystemExit("input.txt is missing. See README.md for how to download it.")
text = data_path.read_text(encoding="utf-8")

# character-level tokenizer: every unique character is one token
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


# first 90% trains, last 10% measures overfitting
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    """Average over many batches, because a single batch's loss is noisy."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # a buffer, not a parameter: the mask is never trained
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs): what each token contains
        q = self.query(x)  # (B, T, hs): what each token is looking for
        # scale by 1/sqrt(head_size) so softmax stays diffuse at initialization
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T)
        # the future may not talk to the past; this is what makes it a decoder
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, hs): what each token communicates
        return wei @ v  # (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, concatenated along the channel dimension."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Per-token computation: each token thinks about what it gathered."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # projection back to the residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Communication then computation, with pre-norm residual connections."""

    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb  # broadcast across the batch
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        # cross_entropy wants the channels second, so flatten batch and time
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # the position table only covers block_size, so crop the context
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def sample(model, num_chars):
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # token 0 is newline
    return decode(model.generate(context, max_new_tokens=num_chars)[0].tolist())


def train(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    first_iter = 0
    best_val = float("inf")
    if args.resume:
        if not state_path.exists():
            raise SystemExit(f"Nothing to resume: {state_path} does not exist.")
        state = torch.load(state_path, map_location=device)
        # the state file holds the latest weights; older ones only had the best
        if "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        optimizer.load_state_dict(state["optimizer"])
        first_iter = state["iter"]
        best_val = state.get("best_val", best_val)
        print(f"resuming from step {first_iter}", flush=True)

    start = time.time()
    for it in range(first_iter, max_iters + 1):
        if it % eval_interval == 0 or it == max_iters:
            losses = estimate_loss(model)
            minutes = (time.time() - start) / 60
            improved = losses["val"] < best_val
            print(
                f"step {it}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f} ({minutes:.1f} min)"
                + (" best" if improved else ""),
                flush=True,
            )
            # keep the weights that generalize best, not the last (most overfit) ones
            if improved:
                best_val = float(losses["val"])
                torch.save(model.state_dict(), checkpoint_path)
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "iter": it, "best_val": best_val},
                state_path,
            )
        if it == max_iters:
            break

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def main():
    model = GPTLanguageModel().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"{args.preset} preset on {device}: {params / 1e6:.2f}M parameters", flush=True)

    if args.generate is not None:
        if not checkpoint_path.exists():
            raise SystemExit(f"No checkpoint at {checkpoint_path}. Train this preset first.")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(sample(model, args.generate))
        return

    train(model)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    text_out = sample(model, 2000)
    sample_path.write_text(text_out, encoding="utf-8")
    print(text_out)
    print(f"\nSaved {checkpoint_path.name} and {sample_path.name}")


if __name__ == "__main__":
    main()
