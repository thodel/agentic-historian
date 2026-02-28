"""
utils/claude_client.py – Claude (text) + Gemini (vision) client with retry & logging
"""
import asyncio
from pathlib import Path
from typing import Optional

import anthropic
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
from PIL import Image

import config

_client: Optional[anthropic.AsyncAnthropic] = None
_gemini_configured = False


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def _configure_gemini():
    global _gemini_configured
    if _gemini_configured:
        return
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env")
    genai.configure(api_key=config.GEMINI_API_KEY)
    _gemini_configured = True


def _get_gemini_model(model: Optional[str] = None) -> genai.GenerativeModel:
    _configure_gemini()
    return genai.GenerativeModel(model or config.GEMINI_VISION_MODEL)


async def _ask_gemini(
    system: str,
    user_text: str,
    image_path: Path,
    model: Optional[str],
    max_tokens: int,
    agent_name: str,
) -> str:
    prompt = f"{system}\n\n{user_text}".strip()

    def _run():
        gemini_model = _get_gemini_model(model)
        with Image.open(image_path) as img:
            response = gemini_model.generate_content(
                [prompt, img],
                generation_config={"max_output_tokens": max_tokens},
            )
        return response.text

    return await asyncio.to_thread(_run)


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
    Send a message to Claude (text) or Gemini (vision) and return the text response.
    If image_path is provided, the request is sent to Gemini.
    """
    if image_path and image_path.exists():
        return await _ask_gemini(system, user_text, image_path, model, max_tokens, agent_name)

    client = get_client()
    model = model or config.CLAUDE_MODEL

    content: list = []
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
