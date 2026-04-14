import os
import re
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, render_template_string, request
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
    text = re.sub(r"@\w+", " <user> ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z0-9<>'!?.,\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_text(text: str, vocab: dict, max_len: int):
    pad_id = vocab["<pad>"]
    unk_id = vocab["<unk>"]
    tokens = clean_text(text).split()
    token_ids = [vocab.get(token, unk_id) for token in tokens][:max_len]
    if not token_ids:
        token_ids = [unk_id]

    length = len(token_ids)
    input_ids = token_ids + [pad_id] * (max_len - length)
    return (
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([length], dtype=torch.long),
    )


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_ids.size(1),
        )
        mask = (
            torch.arange(output.size(1), device=lengths.device)[None, :] < lengths[:, None]
        ).unsqueeze(-1)
        masked_output = output.masked_fill(~mask, 0.0)
        mean_pool = masked_output.sum(dim=1) / lengths.unsqueeze(1).clamp(min=1).float()
        max_pool = output.masked_fill(~mask, float("-inf")).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        features = torch.cat([mean_pool, max_pool], dim=1)
        features = self.dropout(features)
        return self.classifier(features)


def load_artifact(artifact_path: Path):
    artifact = torch.load(artifact_path, map_location="cpu")
    vocab = artifact["vocab"]
    config = artifact["config"]
    raw_label_map = config["label_map"]
    label_map = {int(key): value for key, value in raw_label_map.items()}

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=int(config["embed_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        num_classes=len(label_map),
        dropout=float(config["dropout"]),
        pad_idx=vocab["<pad>"],
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    return model, vocab, int(config["max_len"]), label_map


ARTIFACT_PATH = Path(
    os.environ.get("BILSTM_ARTIFACT_PATH", "artifacts/bilstm_transfer_hatexplain_artifact.pt")
)
if not ARTIFACT_PATH.exists():
    raise FileNotFoundError(
        f"Model artifact not found at {ARTIFACT_PATH}. "
        "Set BILSTM_ARTIFACT_PATH to your saved .pt artifact."
    )

MODEL, VOCAB, MAX_LEN, LABEL_MAP = load_artifact(ARTIFACT_PATH)

app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>BiLSTM Text Classifier</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; }
    textarea { width: 100%; min-height: 140px; padding: 12px; font-size: 16px; }
    button { margin-top: 12px; padding: 10px 18px; font-size: 16px; }
    .card { margin-top: 24px; padding: 18px; border: 1px solid #ddd; border-radius: 10px; }
    .label { font-size: 24px; font-weight: 700; margin-bottom: 12px; }
    .prob { margin: 6px 0; }
    .muted { color: #666; }
  </style>
</head>
<body>
  <h1>BiLSTM Text Classifier</h1>
  <p class="muted">Enter a sentence and the model will classify it as normal, hate speech, or offensive.</p>
  <form method="post">
    <textarea name="text" placeholder="Type a sentence here...">{{ text }}</textarea>
    <br>
    <button type="submit">Analyze</button>
  </form>

  {% if result %}
    <div class="card">
      <div class="label">Prediction: {{ result.predicted_label }}</div>
      {% for label, prob in result.probabilities.items() %}
        <div class="prob">{{ label }}: {{ "%.4f"|format(prob) }}</div>
      {% endfor %}
    </div>
  {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    result = None

    if request.method == "POST":
        text = request.form.get("text", "")
        input_ids, lengths = encode_text(text, VOCAB, MAX_LEN)
        with torch.no_grad():
            logits = MODEL(input_ids, lengths)
            probs = torch.softmax(logits, dim=1)[0].tolist()
        pred_id = int(torch.argmax(logits, dim=1).item())
        result = {
            "predicted_label": LABEL_MAP[pred_id],
            "probabilities": {
                LABEL_MAP[idx]: prob for idx, prob in enumerate(probs)
            },
        }

    return render_template_string(HTML, text=text, result=result)


if __name__ == "__main__":
    app.run(debug=True)
