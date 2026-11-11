import torch
import torch.nn as nn
from torch.nn import functional as F

# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────
batch_size  = 32        # parallel sequences per step
block_size  = 64        # context window (tokens)
max_iters   = 5000      # training steps
eval_interval = 500     # evaluate every N steps
eval_iters  = 200       # batches averaged for eval loss
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd  = 128           # embedding dimension
n_head  = 4             # attention heads
n_layer = 4             # transformer blocks
dropout = 0.1           # regularisation

# Data is loaded explicitly: importing this module never downloads or trains.
from pathlib import Path

train_data = val_data = None
chars = []
stoi = {}
itos = {}
vocab_size = 0


def encode(text):
    unknown = set(text) - set(stoi)
    if unknown:
        raise ValueError(f"Prompt contains characters outside the vocabulary: {sorted(unknown)!r}")
    return [stoi[c] for c in text]


def decode(tokens):
    return ''.join(itos[i] for i in tokens)


def prepare_dataset(path):
    global train_data, val_data, chars, stoi, itos, vocab_size
    text = Path(path).read_text(encoding='utf-8')
    n = int(0.9 * len(text))
    if n <= block_size or len(text) - n <= block_size:
        raise ValueError(f"Both 90/10 splits must contain more than {block_size} characters; use a longer corpus or smaller --block-size")
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = dict(enumerate(chars))
    data = torch.tensor(encode(text), dtype=torch.long)
    train_data, val_data = data[:n], data[n:]


# ── Batch sampler ────────────────────────────
def get_batch(split: str):
    if split not in ('train', 'val'):
        raise ValueError('split must be train or val')
    if train_data is None:
        raise ValueError('Call prepare_dataset before sampling')
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i:i+block_size]   for i in ix])
    y  = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ── Loss estimator (no_grad = fast) ──────────
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ─────────────────────────────────────────────
# MODEL COMPONENTS
# ─────────────────────────────────────────────

class Head(nn.Module):
    """Single causal self-attention head."""

    def __init__(self, head_size: int):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # causal mask — lower-triangular ones
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Scaled dot-product attention scores
        scale = k.shape[-1] ** -0.5
        wei = q @ k.transpose(-2, -1) * scale          # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v   = self.value(x)                            # (B, T, head_size)
        out = wei @ v                                  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """n_head parallel attention heads, results concatenated."""

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise two-layer MLP with ReLU."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One transformer block:
      x → LayerNorm → MultiHeadAttention → residual
        → LayerNorm → FeedForward        → residual
    """

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head          # each head gets equal share
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ff   = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # pre-norm + residual
        x = x + self.ff(self.ln2(x))   # pre-norm + residual
        return x


# ─────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────

class AdamLanguageModel(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(
            *[TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f    = nn.LayerNorm(n_embd)      # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Better weight init (GPT-2 style)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                          # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb                                              # (B, T, n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                           # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Autoregressively generate `max_new_tokens` tokens.
        idx : (B, T) LongTensor of seed tokens
        temperature : >1 → more random, <1 → more focused
        top_k : if set, sample only from top-k logits
        """
        if temperature <= 0 or not __import__('math').isfinite(temperature):
            raise ValueError('temperature must be positive and finite')
        if max_new_tokens < 0 or (top_k is not None and top_k < 1):
            raise ValueError('max_new_tokens must be non-negative and top_k positive')
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError('seed must be a non-empty batch of token sequences')
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]              # trim to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature      # last time-step

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs   = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train(checkpoint="tiny_lm_best.pt"):
    model = AdamLanguageModel(vocab_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f" TinyLM Lab  |  Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Cosine LR scheduler: warms performance in the later stages
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

    best_val_loss = float('inf')

    for step in range(max_iters):

        # ── Forward + backward ───────────────
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ── Periodic evaluation ──────────────
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model)
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"Step {step:>5} | "
                f"train loss: {losses['train']:.4f} | "
                f"val loss: {losses['val']:.4f} | "
                f"lr: {lr_now:.2e}"
            )
            # ── Auto-save best checkpoint ────
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({'model': model.state_dict(), 'chars': chars, 'block_size': block_size, 'n_embd': n_embd, 'n_head': n_head, 'n_layer': n_layer, 'dropout': dropout}, checkpoint)
                print(f"   Best model saved  (val loss {best_val_loss:.4f})")


    print("\n Training complete!")
    return model


# ─────────────────────────────────────────────
# GENERATE SAMPLE TEXT
# ─────────────────────────────────────────────

def generate_text(model, prompt: str = "\n", max_new_tokens: int = 300,
                  temperature: float = 0.8, top_k: int = 40):
    model.eval()
    seed = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    output = model.generate(seed, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=top_k)
    return decode(output[0].tolist())


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    import argparse
    global batch_size, block_size, max_iters, eval_interval, eval_iters, device
    parser = argparse.ArgumentParser(description='TinyLM Lab: explicit local-data training')
    parser.add_argument('--data', required=True, help='Local UTF-8 corpus; no automatic downloads')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--block-size', type=int, default=32)
    parser.add_argument('--eval-iters', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--checkpoint', default='tiny_lm_best.pt')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.block_size, args.eval_iters) < 1:
        parser.error('steps, batch-size, block-size and eval-iters must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available; use --device cpu')
    torch.manual_seed(args.seed)
    batch_size, block_size = args.batch_size, args.block_size
    max_iters, eval_iters, eval_interval = args.steps, args.eval_iters, max(1, args.steps // 5)
    device = args.device
    try:
        prepare_dataset(args.data)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    train(args.checkpoint)


if __name__ == '__main__':
    main()
