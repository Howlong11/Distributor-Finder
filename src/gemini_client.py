from __future__ import annotations

import time
from typing import Any, Dict

import requests

from src.config import AppConfig
from src.utils import extract_json


class GeminiRateLimitError(RuntimeError):
    """Raised when Gemini quota or rate limits prevent request completion."""


class GeminiModelNotFoundError(RuntimeError):
    """Raised when the configured Gemini model name is invalid for the API endpoint."""


class GeminiServiceUnavailableError(RuntimeError):
    """Raised when Gemini is temporarily unavailable or overloaded."""


class GeminiClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.last_response_text = ""
        self.last_response_body: Dict[str, Any] | None = None
        self.last_status_code: int | None = None
        self.last_retry_after: str | None = None

    def generate_json(self, prompt: str, schema_hint: Dict[str, Any]) -> Any:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json",
            },
        }
        if schema_hint:
            payload["generationConfig"]["responseSchema"] = schema_hint

        response_text = self._call_generate_content(payload)
        return extract_json(response_text)

    def generate_text(self, prompt: str, temperature: float = 0.6) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
            },
        }
        return self._call_generate_content(payload).strip()

    def health_check(self) -> Dict[str, Any]:
        try:
            reply = self.generate_text("Reply with exactly: CONNECTED", temperature=0.0)
            return {
                "ok": "CONNECTED" in reply,
                "message": reply,
                "model": self.config.gemini_model,
                "status_code": self.last_status_code,
            }
        except GeminiRateLimitError as exc:
            return {
                "ok": False,
                "message": str(exc),
                "model": self.config.gemini_model,
                "status_code": self.last_status_code,
                "retry_after": self.last_retry_after,
                "raw_body": self.last_response_body,
            }
        except GeminiModelNotFoundError as exc:
            return {
                "ok": False,
                "message": str(exc),
                "model": self.config.gemini_model,
                "status_code": self.last_status_code,
                "retry_after": self.last_retry_after,
                "raw_body": self.last_response_body,
            }
        except GeminiServiceUnavailableError as exc:
            return {
                "ok": False,
                "message": str(exc),
                "model": self.config.gemini_model,
                "status_code": self.last_status_code,
                "retry_after": self.last_retry_after,
                "raw_body": self.last_response_body,
            }
        except Exception as exc:
            return {
                "ok": False,
                "message": str(exc),
                "model": self.config.gemini_model,
                "status_code": self.last_status_code,
                "retry_after": self.last_retry_after,
                "raw_body": self.last_response_body,
            }

    def _call_generate_content(self, payload: Dict[str, Any]) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.gemini_model}:generateContent"
        )
        last_error: requests.HTTPError | None = None

        for attempt in range(1, self.config.gemini_max_retries + 1):
            response = requests.post(
                url,
                params={"key": self.config.gemini_api_key},
                json=payload,
                timeout=45,
            )
            self.last_status_code = response.status_code
            self.last_retry_after = response.headers.get("Retry-After")

            if response.status_code == 404:
                raise GeminiModelNotFoundError(
                    f"Gemini model '{self.config.gemini_model}' was not found for this API endpoint. "
                    "Update GEMINI_MODEL in your .env to a supported model such as 'gemini-2.0-flash'."
                )

            if response.status_code == 503:
                error_detail = self._extract_error_detail(response)
                retry_after = response.headers.get("Retry-After")
                if attempt < self.config.gemini_max_retries:
                    time.sleep(self._retry_delay(attempt, retry_after))
                    continue
                raise GeminiServiceUnavailableError(
                    self._build_service_unavailable_message(retry_after, error_detail)
                )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                error_detail = self._extract_error_detail(response)
                wait_seconds = self._retry_delay(attempt, retry_after)
                last_error = requests.HTTPError(
                    f"Gemini rate limited request with status 429 on attempt {attempt}.",
                    response=response,
                )
                if attempt < self.config.gemini_max_retries:
                    time.sleep(wait_seconds)
                    continue
                raise GeminiRateLimitError(
                    self._build_rate_limit_message(retry_after, error_detail)
                ) from last_error

            response.raise_for_status()
            body = response.json()
            self.last_response_body = body
            candidates = body.get("candidates", [])
            if not candidates:
                raise ValueError("Gemini returned no candidates.")

            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts).strip()
            if not text:
                raise ValueError("Gemini returned an empty response.")
            self.last_response_text = text
            return text

        raise GeminiRateLimitError(
            "Gemini rate limit reached before a valid response was returned."
        ) from last_error

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return max(0.5, float(retry_after))
            except ValueError:
                pass
        return self.config.gemini_retry_delay_seconds * attempt

    def _extract_error_detail(self, response: requests.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.text.strip()

        self.last_response_body = body
        error = body.get("error", {})
        message = error.get("message", "")
        status = error.get("status", "")
        details = []
        if status:
            details.append(status)
        if message:
            details.append(message)
        return " | ".join(details).strip()

    def _build_rate_limit_message(self, retry_after: str | None, error_detail: str) -> str:
        parts = [
            "Gemini rate limit reached.",
            "This often means a temporary requests-per-minute or tokens-per-minute limit, even if total free tokens remain.",
        ]
        if retry_after:
            parts.append(f"Retry-After: {retry_after} seconds.")
        else:
            parts.append("Wait a minute and try again.")
        if error_detail:
            parts.append(f"API detail: {error_detail}")
        return " ".join(parts)

    def _build_service_unavailable_message(
        self, retry_after: str | None, error_detail: str
    ) -> str:
        parts = [
            "Gemini service is temporarily unavailable or the model is overloaded.",
            "Your API key may still be valid; this is usually a temporary model-side issue.",
        ]
        if retry_after:
            parts.append(f"Retry-After: {retry_after} seconds.")
        else:
            parts.append("Try again shortly or switch to a more stable model.")
        if error_detail:
            parts.append(f"API detail: {error_detail}")
        return " ".join(parts)
