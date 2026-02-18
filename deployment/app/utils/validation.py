import json

from app.data.preprocessing import build_final_text, detect_language, is_empty_text
from app.utils.errors import InputError


class InputValidator:
    def __init__(self, config):
        self.config = config

    def validate(self, tweet, keyword=None):
        warnings = []
        final_text = build_final_text(tweet, keyword=keyword, config=self.config)
        if is_empty_text(final_text):
            raise InputError(
                "EMPTY_TWEET",
                "Tweet became empty after cleaning. Provide valid text.",
            )

        tokens = final_text.split()
        if len(tokens) < 3:
            warnings.append(self._warning(
                "SHORT_INPUT",
                "Input is very short; prediction may be unreliable.",
                {"token_count": len(tokens)},
            ))
        if len(tokens) > self.config.MAX_LENGTH:
            warnings.append(self._warning(
                "TRIMMED_LENGTH",
                "Input was trimmed to max length.",
                {"token_count": len(tokens), "max_length": self.config.MAX_LENGTH},
            ))

        lang, prob = detect_language(final_text)
        if lang is None:
            warnings.append(self._warning(
                "LANGUAGE_UNDETECTED",
                "Could not confidently detect input language.",
            ))
        elif lang != "en":
            token_count = len(tokens)
            enough_text = token_count >= 6 and len(final_text) >= 30
            strong_confidence = prob is not None and prob >= 0.95

            if enough_text and strong_confidence:
                raise InputError(
                    "NON_ENGLISH",
                    "Detected non-English input. Model is trained on English.",
                    {"lang": lang, "prob": round(prob, 2)},
                )

            warnings.append(self._warning(
                "LANGUAGE_SUSPECT",
                "Language detection was uncertain; continuing with prediction.",
                {"lang": lang, "prob": round(prob, 2) if prob is not None else None},
            ))

        return final_text, warnings

    def warnings_json(self, warnings):
        return [json.dumps(w, ensure_ascii=True) for w in warnings]

    def _warning(self, warning_code, message, details=None):
        payload = {
            "warning_code": warning_code,
            "message": message,
        }
        if details:
            payload["details"] = details
        return payload
