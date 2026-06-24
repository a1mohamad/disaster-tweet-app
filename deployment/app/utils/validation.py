import json
from typing import Any

from app.data.preprocessing import build_final_text, detect_language, is_empty_text
from app.utils.errors import InputError


class InputValidator:
    """Validate and normalize user input before inference."""

    def __init__(self, config: object) -> None:
        self.config = config

    def validate(
        self, tweet: str, keyword: str | None = None
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return model-ready text plus warnings, or raise an input error."""
        warnings = []
        final_text = build_final_text(tweet, keyword=keyword, config=self.config)
        if is_empty_text(final_text):
            raise InputError(
                "EMPTY_TWEET",
                "Tweet became empty after cleaning. Provide valid text.",
            )

        tokens = final_text.split()
        # Very short inputs can produce valid logits but low-confidence semantics.
        if len(tokens) < 3:
            warnings.append(self._warning(
                "SHORT_INPUT",
                "Input is very short; prediction may be unreliable.",
                {"token_count": len(tokens)},
            ))
        # The model only consumes MAX_LENGTH tokens; tell callers when text is clipped.
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
            # langdetect is noisy on tiny tweets, so only hard-fail longer confident cases.
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

    def warnings_json(self, warnings: list[dict[str, Any]]) -> list[str]:
        """Serialize warning dictionaries for command-line output."""
        return [json.dumps(w, ensure_ascii=True) for w in warnings]

    def _warning(
        self,
        warning_code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a consistent warning payload."""
        payload = {
            "warning_code": warning_code,
            "message": message,
        }
        if details:
            payload["details"] = details
        return payload
