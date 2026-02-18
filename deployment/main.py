import argparse
import json
import sys

from app.utils.seed import seed_everything
from app.app_config import AppConfig
from app.model.predictor import DisasterTwittsPredictor
from app.data.preprocessing import (load_vocabs, 
                                    tokenize_and_pad)
from app.utils.validation import InputValidator
from app.utils.errors import AppError, ArtifactError

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, default="")
    parser.add_argument("--tweet", type=str, default="")
    parser.add_argument("--use-label", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_args()
        config = AppConfig()
        seed_everything(config.SEED)

        if not config.VOCAB_PATH.exists():
            raise ArtifactError(
                "VOCAB_NOT_FOUND",
                "Vocabulary file not found.",
                {"path": str(config.VOCAB_PATH)},
            )
        if not config.MODEL_PATH.exists():
            raise ArtifactError(
                "MODEL_NOT_FOUND",
                "Model file not found.",
                {"path": str(config.MODEL_PATH)},
            )
        if args.use_label and not config.LABEL_MAPPING_PATH.exists():
            raise ArtifactError(
                "LABEL_MAPPING_NOT_FOUND",
                "Label mapping file not found.",
                {"path": str(config.LABEL_MAPPING_PATH)},
            )

        word2idx, idx2word, vocab_size = load_vocabs(config.VOCAB_PATH)

        def get_valid_inputs():
            validator = InputValidator(config)
            while True:
                keyword = args.keyword or str(input("Keyword: "))
                twitt = args.tweet or str(input("Tweet: "))
                try:
                    final_text, warnings = validator.validate(twitt, keyword=keyword)
                except AppError as exc:
                    print(exc.to_json())
                    if args.tweet:
                        sys.exit(1)
                    continue
                for w_json in validator.warnings_json(warnings):
                    print(w_json)
                return final_text

        final_text = get_valid_inputs()
        input_ids, input_length = tokenize_and_pad(final_text, word2idx, config)
        predictor = DisasterTwittsPredictor.from_config(config, vocab_size)
        for w in predictor.warnings:
            print(json.dumps(w, ensure_ascii=True))
        if args.use_label:
            predictor.load_label_mapping(config.LABEL_MAPPING_PATH)
            prob, label, label_name = predictor.predict(
                input_ids, input_length, return_label_name=True
                )
            print(
                f"Probability: {prob:.4f} | Label: {label_name} | Threshold: {predictor.threshold:.4f}"
                )
        else:
            prob, label = predictor.predict(
                input_ids, input_length, return_label_name=False
                )
            print(
                f"Probability: {prob:.4f} | Label: {label} | Threshold: {predictor.threshold:.4f}"
                )
    except AppError as exc:
        print(exc.to_json())
        sys.exit(1)
