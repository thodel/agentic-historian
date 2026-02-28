"""
config.py – Central configuration for Agentic Historian
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
DATA_DIR          = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
HOT_FOLDER        = Path(os.getenv("HOT_FOLDER",         DATA_DIR / "hot_folder"))
OUTPUT_DIR        = Path(os.getenv("OUTPUT_DIR",         DATA_DIR / "outputs"))
CORPUS_DIR        = Path(os.getenv("CORPUS_DIR",         DATA_DIR / "corpus"))
TRANSCRIPTION_DIR = Path(os.getenv("TRANSCRIPTION_DIR", DATA_DIR / "transcriptions"))
DESCRIPTION_DIR   = Path(os.getenv("DESCRIPTION_DIR",   DATA_DIR / "descriptions"))
KNOWLEDGE_HUB_DIR = Path(os.getenv("KNOWLEDGE_HUB_DIR", BASE_DIR / "knowledge_hub" / "data"))
TMP_DIR           = Path(os.getenv("TMP_DIR",           DATA_DIR / "tmp"))

for _p in [HOT_FOLDER, OUTPUT_DIR, CORPUS_DIR, TRANSCRIPTION_DIR,
           DESCRIPTION_DIR, KNOWLEDGE_HUB_DIR, TMP_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# ── Discord ────────────────────────────────────────────────────────────────
DISCORD_BOT_TOKEN          = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_GUILD_ID           = int(os.getenv("DISCORD_GUILD_ID", 0)) or None
DISCORD_CHANNEL_HISTORIAN  = os.getenv("DISCORD_CHANNEL_HISTORIAN", "historian-general")

# ── Anthropic ──────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL        = os.getenv("CLAUDE_MODEL",        "claude-opus-4-5")

# ── Gemini (Vision) ────────────────────────────────────────────────────────
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-3.1-pro")

# ── HuggingFace ────────────────────────────────────────────────────────────
HF_TOKEN   = os.getenv("HF_TOKEN",   "")
HF_REPO_ID = os.getenv("HF_REPO_ID", "")

# ── GitHub ─────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO  = os.getenv("GITHUB_REPO",  "")

# ── Voyant ─────────────────────────────────────────────────────────────────
VOYANT_URL = os.getenv("VOYANT_URL", "https://voyant-tools.org")

# ── Agent settings ─────────────────────────────────────────────────────────
HTR_QUALITY_THRESHOLD = float(os.getenv("HTR_QUALITY_THRESHOLD", "0.75"))
MAX_RETRIES           = int(os.getenv("MAX_RETRIES", "3"))
AGENT_LOG_TOKENS      = os.getenv("AGENT_LOG_TOKENS", "true").lower() == "true"

# ── Feature flags ──────────────────────────────────────────────────────────
ENABLE_HOT_FOLDER_WATCH = os.getenv("ENABLE_HOT_FOLDER_WATCH", "true").lower() == "true"
ENABLE_HF_UPLOAD        = os.getenv("ENABLE_HF_UPLOAD",        "true").lower() == "true"
ENABLE_GITHUB_PUSH      = os.getenv("ENABLE_GITHUB_PUSH",      "false").lower() == "true"

# ── Supported file extensions ──────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
PDF_EXTENSIONS   = {".pdf"}
TEXT_EXTENSIONS  = {".txt", ".md"}
ALL_EXTENSIONS   = IMAGE_EXTENSIONS | PDF_EXTENSIONS | TEXT_EXTENSIONS
