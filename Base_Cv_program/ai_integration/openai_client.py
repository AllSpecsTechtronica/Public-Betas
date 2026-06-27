"""Environment-only OpenAI integration for Base CV.

This module intentionally contains no hard-coded credentials. Set
`OPENAI_API_KEY` in your shell, an ignored `.env` file, or a deployment secret
manager when cloud analysis is needed.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Optional


class OpenAIClient:
    """Small synchronous image-analysis wrapper used by the legacy Base CV UI."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        self.model = (model or os.environ.get("BASE_CV_OPENAI_MODEL") or "gpt-4.1-mini").strip()
        self._client: Any = None
        self._analysis_count = 0
        if self.api_key:
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(api_key=self.api_key)
            except Exception:
                self._client = None

    def is_available(self) -> bool:
        return self._client is not None

    def analyze_image(self, image: Any, prompt: Optional[str] = None) -> Optional[str]:
        if self._client is None:
            return None
        image_b64 = self._encode_image(image)
        if not image_b64:
            return None
        request_prompt = prompt or "Analyze this image and describe the visible objects and notable details."
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": request_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=600,
            )
            self._analysis_count += 1
            choice = response.choices[0] if response.choices else None
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None)
            return str(content).strip() if content else None
        except Exception:
            return None

    def get_session_info(self) -> dict[str, Any]:
        return {
            "provider": "openai",
            "model": self.model,
            "available": self.is_available(),
            "analyses": self._analysis_count,
        }

    def save_session_report(self) -> None:
        return None

    @staticmethod
    def _encode_image(image: Any) -> str:
        if image is None:
            return ""
        if isinstance(image, str):
            return image
        if isinstance(image, bytes):
            return base64.b64encode(image).decode("ascii")
        try:
            import cv2  # type: ignore

            ok, encoded = cv2.imencode(".jpg", image)
            if not ok:
                return ""
            return base64.b64encode(encoded.tobytes()).decode("ascii")
        except Exception:
            return ""


openai_client = OpenAIClient()

__all__ = ["OpenAIClient", "openai_client"]
