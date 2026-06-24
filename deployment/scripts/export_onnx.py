import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.app_config import AppConfig
from app.data.preprocessing import build_final_text, load_vocabs, tokenize_and_pad
from app.model.disaster_model import DisasterTwittsClassifier


SAMPLE_TEXTS = [
    ("", "Forest fire near homes, residents ordered to evacuate immediately."),
    ("weather", "Heavy rain downtown but traffic is moving normally."),
]


def load_threshold(config: object) -> float:
    """Load the trained threshold, falling back to the configured default."""
    try:
        threshold = torch.load(config.THRESHOLD_PATH, map_location="cpu")
        if isinstance(threshold, torch.Tensor):
            threshold = threshold.item()
        return float(threshold)
    except FileNotFoundError:
        return float(config.THRESHOLD)


def sigmoid(value: float) -> float:
    """Compute sigmoid as a plain Python float."""
    return float(1.0 / (1.0 + np.exp(-float(value))))


def export_onnx(
    config: object, vocab_size: int, output_path: Path
) -> DisasterTwittsClassifier:
    """Export the trained PyTorch model to ONNX format."""
    model = DisasterTwittsClassifier.from_pretrained(config, vocab_size)
    dummy_ids = torch.zeros((1, config.MAX_LENGTH), dtype=torch.long)
    dummy_length = torch.ones((1,), dtype=torch.long)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_ids, dummy_length),
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids", "input_length"],
        output_names=["logits"],
        dynamo=False,
        dynamic_axes={
            "input_ids": {0: "batch"},
            "input_length": {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    return model


def validate_onnx(
    config: object,
    model: DisasterTwittsClassifier,
    word2idx: dict[str, int],
    output_path: Path,
    tolerance: float,
) -> tuple[list[dict[str, Any]], float]:
    """Compare ONNX and PyTorch probabilities on representative samples."""
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])

    max_delta = 0.0
    results = []
    for keyword, tweet in SAMPLE_TEXTS:
        final_text = build_final_text(tweet, keyword=keyword, config=config)
        input_ids, input_length = tokenize_and_pad(final_text, word2idx, config)

        torch_ids = torch.as_tensor(input_ids, dtype=torch.long, device=model.device)
        torch_length = torch.as_tensor(input_length, dtype=torch.long)
        with torch.no_grad():
            torch_logit = model(torch_ids, lengths=torch_length).detach().cpu().item()

        onnx_logit = session.run(
            ["logits"],
            {
                "input_ids": input_ids.astype(np.int64),
                "input_length": input_length.astype(np.int64),
            },
        )[0].reshape(-1)[0]

        torch_prob = sigmoid(torch_logit)
        onnx_prob = sigmoid(onnx_logit)
        delta = abs(torch_prob - onnx_prob)
        max_delta = max(max_delta, delta)
        results.append(
            {
                "keyword": keyword,
                "tweet": tweet,
                "torch_probability": torch_prob,
                "onnx_probability": onnx_prob,
                "delta": delta,
            }
        )

    if max_delta > tolerance:
        raise RuntimeError(
            f"ONNX validation failed: max probability delta {max_delta:.8f} > {tolerance}"
        )
    return results, max_delta


def main() -> None:
    """Export artifacts and optionally validate ONNX parity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    config = AppConfig()
    output_path = args.output or config.ONNX_MODEL_PATH

    word2idx, idx2word, vocab_size = load_vocabs(config.VOCAB_PATH)
    model = export_onnx(config, vocab_size, output_path)

    threshold = load_threshold(config)
    config.THRESHOLD_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.THRESHOLD_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold}, f, indent=2)
        f.write("\n")

    payload = {
        "onnx_path": str(output_path),
        "threshold_json_path": str(config.THRESHOLD_JSON_PATH),
        "threshold": threshold,
    }

    if not args.skip_validation:
        results, max_delta = validate_onnx(
            config, model, word2idx, output_path, args.tolerance
        )
        payload["validation"] = {
            "max_probability_delta": max_delta,
            "samples": results,
        }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
