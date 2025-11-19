import numpy as np
import torch
from pathlib import Path
import sys

"""Ensure project root is on sys.path when running this file directly."""
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import TrainingConfig as CFG
from model.model import TinyESPNet


def main() -> None:
    cfg = CFG()
    device = torch.device("cpu")

    pcm_path = PROJECT_ROOT / "demo" / "input.pcm"
    with open(pcm_path, "rb") as f:
        audio = np.frombuffer(f.read(), dtype="<i2").astype(np.float32) / 32768.0

    if audio.size < cfg.SEGMENT_SAMPLES:
        audio = np.pad(audio, (0, cfg.SEGMENT_SAMPLES - audio.size))
    else:
        audio = audio[: cfg.SEGMENT_SAMPLES]

    model = TinyESPNet(
        num_classes=2,
        base_channels=getattr(cfg, "base_channels", 12),
        classifier_hidden=getattr(cfg, "classifier_hidden", 24),
        dropout=getattr(cfg, "model_dropout", 0.0),
    ).to(device)

    # Resolve weights path robustly: if relative, treat it as relative to project root
    weights_path = cfg.MODEL_EXPORT
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        tensor = torch.from_numpy(audio).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    print("probabilities:", probs.tolist())


if __name__ == "__main__":
    main()

