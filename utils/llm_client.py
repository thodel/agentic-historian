"""
utils/llm_client.py – Gemini (text + vision) client with retry & logging
"""
import asyncio
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
from PIL import Image

import config

_gemini_configured = False


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
    return genai.GenerativeModel(model or config.GEMINI_TEXT_MODEL)


def _build_prompt(system: str, user_text: str) -> str:
    if system:
        return f"{system}\n\n{user_text}".strip()
    return user_text


async def _ask_gemini(
    prompt: str,
    image_path: Optional[Path],
    model: Optional[str],
    max_tokens: int,
) -> str:

    def _run():
        gemini_model = _get_gemini_model(model)
        if image_path and image_path.exists():
            with Image.open(image_path) as img:
                response = gemini_model.generate_content(
                    [prompt, img],
                    generation_config={"max_output_tokens": max_tokens},
                )
        else:
            response = gemini_model.generate_content(
                prompt,
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
    Send a message to Gemini (text or vision) and return the text response.
    """
    prompt = _build_prompt(system, user_text)

    selected_model = model
    if not selected_model:
        selected_model = (
            config.GEMINI_VISION_MODEL if image_path else config.GEMINI_TEXT_MODEL
        )

    response_text = await _ask_gemini(
        prompt,
        image_path=image_path if image_path and image_path.exists() else None,
        model=selected_model,
        max_tokens=max_tokens,
    )

    if config.AGENT_LOG_TOKENS:
        logger.info(f"[{agent_name}] Gemini request completed.")

    return response_text


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
