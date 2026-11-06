import random
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datasets import load_dataset
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

random.seed(0)
torch.manual_seed(0)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = Tee(sys.__stdout__, open("results.txt", "w", encoding="utf-8"))

N_PAIRS = 2000
MAX_WORDS = 6
EPOCHS = 30
BATCH = 64
HIDDEN = 256
EMB = 128

print("=== Задание 1. Подготовка данных ===")


def tokenize(s):
    return re.findall(r"\w+|[^\w\s]", s.lower())


stream = load_dataset("Helsinki-NLP/opus-100", "en-ru", split="train", streaming=True)
pairs = []
for row in stream:
    en, ru = tokenize(row["translation"]["en"]), tokenize(row["translation"]["ru"])
    if 2 <= len(en) <= MAX_WORDS and 2 <= len(ru) <= MAX_WORDS:
        pairs.append((en, ru))
    if len(pairs) >= N_PAIRS:
        break

random.shuffle(pairs)
split = int(len(pairs) * 0.9)
train_pairs, test_pairs = pairs[:split], pairs[split:]
print(f"Всего пар: {len(pairs)}, обучение: {len(train_pairs)}, тест: {len(test_pairs)}")
for en, ru in pairs[:5]:
    print(" ".join(en), "->", " ".join(ru))

PAD, SOS, EOS, UNK = 0, 1, 2, 3


def build_vocab(sents):
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for s in sents:
        for t in s:
            vocab.setdefault(t, len(vocab))
    return vocab


src_vocab = build_vocab(p[0] for p in train_pairs)
tgt_vocab = build_vocab(p[1] for p in train_pairs)
tgt_itos = {i: t for t, i in tgt_vocab.items()}
print(f"Словарь EN: {len(src_vocab)}, словарь RU: {len(tgt_vocab)}")

SRC_LEN = MAX_WORDS + 1
TGT_LEN = MAX_WORDS + 2


def encode(tokens, vocab, length, sos=False):
    ids = ([SOS] if sos else []) + [vocab.get(t, UNK) for t in tokens] + [EOS]
    return ids + [PAD] * (length - len(ids))


def make_tensors(data):
    src = torch.tensor([encode(p[0], src_vocab, SRC_LEN) for p in data])
    tgt = torch.tensor([encode(p[1], tgt_vocab, TGT_LEN, sos=True) for p in data])
    return src, tgt


src_train, tgt_train = make_tensors(train_pairs)
print("Пример после паддинга:", src_train[0].tolist(), tgt_train[0].tolist())


class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB, padding_idx=PAD)
        self.gru = nn.GRU(EMB, HIDDEN, batch_first=True)

    def forward(self, x):
        return self.gru(self.emb(x))


class BahdanauAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(HIDDEN, HIDDEN)
        self.U = nn.Linear(HIDDEN, HIDDEN)
        self.v = nn.Linear(HIDDEN, 1)

    def forward(self, hidden, enc_out, mask):
        scores = self.v(torch.tanh(self.W(hidden).unsqueeze(1) + self.U(enc_out))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, attention):
        super().__init__()
        self.attention = BahdanauAttention() if attention else None
        self.emb = nn.Embedding(vocab_size, EMB, padding_idx=PAD)
        self.gru = nn.GRUCell(EMB + (HIDDEN if attention else 0), HIDDEN)
        self.out = nn.Linear(HIDDEN * (2 if attention else 1), vocab_size)

    def forward(self, token, hidden, enc_out, mask):
        e = self.emb(token)
        weights = None
        if self.attention:
            context, weights = self.attention(hidden, enc_out, mask)
            hidden = self.gru(torch.cat([e, context], -1), hidden)
            logits = self.out(torch.cat([hidden, context], -1))
        else:
            hidden = self.gru(e, hidden)
            logits = self.out(hidden)
        return logits, hidden, weights


class Seq2Seq(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.encoder = Encoder(len(src_vocab))
        self.decoder = Decoder(len(tgt_vocab), attention)

    def forward(self, src, tgt):
        enc_out, h = self.encoder(src)
        hidden, mask = h.squeeze(0), src != PAD
        logits = []
        for t in range(tgt.size(1) - 1):
            step, hidden, _ = self.decoder(tgt[:, t], hidden, enc_out, mask)
            logits.append(step)
        return torch.stack(logits, 1)

    @torch.no_grad()
    def translate(self, tokens):
        src = torch.tensor([encode(tokens, src_vocab, SRC_LEN)])
        enc_out, h = self.encoder(src)
        hidden, mask = h.squeeze(0), src != PAD
        token, result, attn = torch.tensor([SOS]), [], []
        for _ in range(TGT_LEN):
            logits, hidden, w = self.decoder(token, hidden, enc_out, mask)
            token = logits.argmax(-1)
            if token.item() == EOS:
                break
            result.append(tgt_itos[token.item()])
            if w is not None:
                attn.append(w[0, :len(tokens) + 1].tolist())
        return result, attn


def train(model, name):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    losses = []
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(src_train))
        total = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            src, tgt = src_train[idx], tgt_train[idx]
            opt.zero_grad()
            out = model(src, tgt)
            loss = loss_fn(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        losses.append(total / len(perm))
        if epoch % 5 == 0:
            print(f"[{name}] эпоха {epoch}/{EPOCHS}, loss = {losses[-1]:.4f}")
    return losses, time.time() - start


def bleu(model, data):
    model.eval()
    refs = [[p[1]] for p in data]
    hyps = [model.translate(p[0])[0] for p in data]
    return corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)


def bleu_on_train(model):
    return bleu(model, train_pairs[:200])


print("\n=== Задание 2. Seq2Seq без внимания ===")
base = Seq2Seq(attention=False)
base_losses, base_time = train(base, "Seq2Seq")

print("\n=== Задание 3. Seq2Seq + Bahdanau Attention ===")
attn = Seq2Seq(attention=True)
attn_losses, attn_time = train(attn, "Seq2Seq+Attention")

print("\n=== Сравнение моделей ===")
rows = [("Seq2Seq", base, base_losses, base_time), ("Seq2Seq+Attention", attn, attn_losses, attn_time)]
print(f"{'Модель':<20}{'Final loss':>12}{'BLEU train':>12}{'BLEU test':>12}{'Время, c':>10}")
for name, m, l, t in rows:
    print(f"{name:<20}{l[-1]:>12.4f}{bleu_on_train(m):>12.4f}{bleu(m, test_pairs):>12.4f}{t:>10.1f}")

print("\nПримеры переводов (тест):")
for en, ru in test_pairs[:5]:
    print("EN:", " ".join(en))
    print("  эталон:      ", " ".join(ru))
    print("  без внимания:", " ".join(base.translate(en)[0]))
    print("  с вниманием: ", " ".join(attn.translate(en)[0]))

print("\n=== Задание 4. Визуализация ===")
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), base_losses, label="Seq2Seq")
plt.plot(range(1, EPOCHS + 1), attn_losses, label="Seq2Seq + Attention")
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.title("Изменение функции потерь при обучении")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss.png", dpi=120)
print("Сохранён график loss.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (en, _) in zip(axes, train_pairs[:3]):
    words, weights = attn.translate(en)
    if not weights:
        continue
    ax.imshow(weights, cmap="viridis")
    ax.set_xticks(range(len(en) + 1))
    ax.set_xticklabels(en + ["<eos>"], rotation=45)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_title(" ".join(en))
plt.suptitle("Attention weights (Bahdanau)")
plt.tight_layout()
plt.savefig("attention.png", dpi=120)
print("Сохранена тепловая карта attention.png")
