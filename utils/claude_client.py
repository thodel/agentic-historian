"""
utils/claude_client.py – Shared Anthropic client with retry & token logging
"""
import asyncio
import base64
from pathlib import Path
from typing import Optional

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

import config

_client: Optional[anthropic.AsyncAnthropic] = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def _encode_image(path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image file."""
    suffix = path.suffix.lower()
    media_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".tif": "image/tiff", ".tiff": "image/tiff",
        ".webp": "image/webp",
    }
    media_type = media_map.get(suffix, "image/jpeg")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def ask(
    system: str,
    user_text: str,
    image_path: Optional[Path] = None,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    agent_name: str = "unknown",
) -> str:
    """
    Send a message to Claude and return the text response.
    Optionally attach an image for vision tasks.
    """
    client = get_client()
    model = model or config.CLAUDE_MODEL

    content: list = []
    if image_path and image_path.exists():
        img_data, media_type = _encode_image(image_path)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": img_data},
        })
    content.append({"type": "text", "text": user_text})

    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
    )

    if config.AGENT_LOG_TOKENS:
        usage = response.usage
        logger.info(
            f"[{agent_name}] tokens – input: {usage.input_tokens}, "
            f"output: {usage.output_tokens}"
        )

    return response.content[0].text


async def ask_structured(
    system: str,
    user_text: str,
    image_path: Optional[Path] = None,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    agent_name: str = "unknown",
) -> str:
    """Same as ask() but explicitly requests JSON output."""
    json_system = (
        system
        + "\n\nALWAYS respond with valid JSON only. No markdown, no preamble."
    )
    return await ask(json_system, user_text, image_path, model, max_tokens, agent_name)
