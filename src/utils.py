from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Expected JSON text but received an empty response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1).strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("The model response did not contain valid JSON.")
