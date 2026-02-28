"""
bot.py – Discord Bot for Agentic Historian
Slash commands control the entire agent pipeline.

Commands:
  /process <attachment>          Full pipeline for one document
  /agent a|b|c|d|e <...>        Run a specific agent
  /corpus [name] [doc_ids]       Run corpus analysis (Agent D)
  /report                        Meta-agent report (Agent E)
  /hot_folder                    Process everything in hot folder
  /hub list|add_person|add_place|add_keyword|add_type  Manage Knowledge Hub
  /status                        Show system status
  /help                          Show command overview
"""
import asyncio
import io
import json
import os
from pathlib import Path
from typing import Optional

import discord
from discord import app_commands
from loguru import logger

import config
from orchestrator import Orchestrator

# ── Bot setup ──────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# Global orchestrator (initialised once bot is ready)
orc: Optional[Orchestrator] = None


# ── Helpers ────────────────────────────────────────────────────────────────

async def _send_long(
    interaction: discord.Interaction,
    text: str,
    ephemeral: bool = False,
):
    """Split long messages into chunks ≤ 2000 chars."""
    chunks = [text[i:i+1900] for i in range(0, len(text), 1900)]
    for i, chunk in enumerate(chunks):
        if i == 0:
            await interaction.followup.send(chunk, ephemeral=ephemeral)
        else:
            await interaction.channel.send(chunk)


async def _save_attachment(attachment: discord.Attachment) -> Path:
    """Download a Discord attachment to the hot folder."""
    dest = config.HOT_FOLDER / attachment.filename
    data = await attachment.read()
    dest.write_bytes(data)
    return dest


def _make_status_callback(interaction: discord.Interaction):
    """Returns an async callback that sends progress messages."""
    async def callback(msg: str):
        try:
            await interaction.channel.send(msg)
        except Exception:
            pass
    return callback


# ── /process ──────────────────────────────────────────────────────────────

@tree.command(name="process", description="Run full pipeline on a document image/PDF")
@app_commands.describe(
    file="Image or PDF of the manuscript",
    corpus="Also run corpus analysis after processing? (yes/no)",
)
async def cmd_process(
    interaction: discord.Interaction,
    file: discord.Attachment,
    corpus: str = "no",
):
    await interaction.response.defer(thinking=True)
    run_corpus = corpus.lower() in {"yes", "ja", "y", "1", "true"}

    image_path = await _save_attachment(file)
    cb = _make_status_callback(interaction)
    orc._cb = cb

    result = await orc.run_full_pipeline(image_path, run_corpus=run_corpus)

    # Build summary message
    parts = [f"# ✅ Pipeline complete: `{result.doc_id}`\n"]

    if result.htr:
        parts.append(orc.agent_a.format_discord_summary(result.htr))
    if result.description:
        parts.append(orc.agent_b.format_discord_summary(result.description))
    if result.entities:
        parts.append(orc.agent_c.format_discord_summary(result.entities))
    if result.corpus:
        parts.append(orc.agent_d.format_discord_summary(result.corpus))
    if result.errors:
        parts.append("### ⚠️ Errors\n" + "\n".join(f"- {e}" for e in result.errors))

    await _send_long(interaction, "\n\n".join(parts))


# ── /agent ────────────────────────────────────────────────────────────────

@tree.command(name="agent", description="Run a specific agent (a/b/c/d/e)")
@app_commands.describe(
    agent="Agent to run: a (HTR), b (description), c (entities), d (corpus), e (meta)",
    doc_id="Document ID (stem of .txt file, e.g. 'missive_001')",
    file="Image file – required for agent a and b",
    corpus_name="Corpus name – for agent d",
)
async def cmd_agent(
    interaction: discord.Interaction,
    agent: str,
    doc_id: str = "",
    file: Optional[discord.Attachment] = None,
    corpus_name: str = "main",
):
    await interaction.response.defer(thinking=True)
    agent = agent.lower().strip()
    image_path = None

    if file:
        image_path = await _save_attachment(file)
        doc_id = doc_id or image_path.stem

    try:
        if agent == "a":
            if not image_path:
                await interaction.followup.send("❌ Agent A requires a file attachment.")
                return
            result = await orc.run_agent_a(image_path)
            msg = orc.agent_a.format_discord_summary(result)

        elif agent == "b":
            if not doc_id:
                await interaction.followup.send("❌ Provide doc_id or a file.")
                return
            result = await orc.run_agent_b(doc_id, image_path)
            msg = orc.agent_b.format_discord_summary(result) if result else "❌ No transcription found."

        elif agent == "c":
            if not doc_id:
                await interaction.followup.send("❌ Provide doc_id.")
                return
            result = await orc.run_agent_c(doc_id)
            msg = orc.agent_c.format_discord_summary(result) if result else "❌ No transcription found."

        elif agent == "d":
            result = await orc.run_agent_d(corpus_name=corpus_name)
            msg = orc.agent_d.format_discord_summary(result) if result else "❌ No transcriptions found."

        elif agent == "e":
            result = await orc.run_agent_e()
            msg = orc.agent_e.format_discord_summary(result)

        else:
            msg = f"❌ Unknown agent: `{agent}`. Use a, b, c, d or e."

    except Exception as e:
        msg = f"❌ Error running agent `{agent}`: {e}"
        logger.exception(e)

    await _send_long(interaction, msg)


# ── /corpus ───────────────────────────────────────────────────────────────

@tree.command(name="corpus", description="Run corpus analysis (Agent D) on all or selected docs")
@app_commands.describe(
    name="Name for this corpus (default: main)",
    doc_ids="Comma-separated list of doc IDs to include (default: all)",
)
async def cmd_corpus(
    interaction: discord.Interaction,
    name: str = "main",
    doc_ids: str = "",
):
    await interaction.response.defer(thinking=True)
    ids = [d.strip() for d in doc_ids.split(",") if d.strip()] or None
    result = await orc.run_agent_d(corpus_name=name, doc_ids=ids)
    if result:
        await _send_long(interaction, orc.agent_d.format_discord_summary(result))
    else:
        await interaction.followup.send("❌ Corpus analysis failed – no transcriptions found.")


# ── /report ───────────────────────────────────────────────────────────────

@tree.command(name="report", description="Generate Meta-Agent report (Agent E)")
async def cmd_report(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    report = await orc.run_agent_e()
    await _send_long(interaction, orc.agent_e.format_discord_summary(report))


# ── /hot_folder ───────────────────────────────────────────────────────────

@tree.command(name="hot_folder", description="Process all files waiting in the hot folder")
async def cmd_hot_folder(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    cb = _make_status_callback(interaction)
    orc._cb = cb
    results = await orc.process_hot_folder()
    if not results:
        await interaction.followup.send("📂 Hot folder was empty – nothing to process.")
        return
    summary = f"# Hot Folder – {len(results)} document(s) processed\n"
    for r in results:
        status = "✅" if not r.errors else "⚠️"
        summary += f"{status} `{r.doc_id}`"
        if r.errors:
            summary += f" – {'; '.join(r.errors)}"
        summary += "\n"
    await _send_long(interaction, summary)


# ── /hub ──────────────────────────────────────────────────────────────────

hub_group = app_commands.Group(name="hub", description="Manage the Knowledge Hub")

@hub_group.command(name="list", description="Show Knowledge Hub contents")
async def hub_list(interaction: discord.Interaction):
    await interaction.response.send_message(orc.hub_summary())

@hub_group.command(name="add_keyword", description="Add a term to the controlled vocabulary")
@app_commands.describe(term="Term to add (e.g. 'arme lüt')")
async def hub_add_keyword(interaction: discord.Interaction, term: str):
    orc.hub_add_keyword(term)
    await interaction.response.send_message(f"✅ Added keyword: `{term}`")

@hub_group.command(name="add_type", description="Add a document type to the hub")
@app_commands.describe(dtype="Document type name (e.g. 'Verhörprotokoll')")
async def hub_add_type(interaction: discord.Interaction, dtype: str):
    orc.hub_add_document_type(dtype)
    await interaction.response.send_message(f"✅ Added document type: `{dtype}`")

@hub_group.command(name="add_person", description="Add a person to the Knowledge Hub")
@app_commands.describe(
    name="Full name",
    variants="Alternative name forms, comma-separated",
    role="Role/office (e.g. Vogt, Schultheiss)",
    location="City/region",
    period="Active period (e.g. 1430–1450)",
    wikidata="Wikidata Q-ID (optional)",
    gnd="GND identifier (optional)",
)
async def hub_add_person(
    interaction: discord.Interaction,
    name: str,
    variants: str = "",
    role: str = "",
    location: str = "",
    period: str = "",
    wikidata: str = "",
    gnd: str = "",
):
    persons = orc.hub.get_persons()
    new_id = f"hub_p_{len(persons)+1:04d}"
    person = {
        "id": new_id,
        "name": name,
        "variants": [v.strip() for v in variants.split(",") if v.strip()],
        "role": role,
        "location": location,
        "active_period": period,
        "gnd_id": gnd or None,
        "wikidata_id": wikidata or None,
        "notes": "",
    }
    orc.hub_add_person(person)
    await interaction.response.send_message(
        f"✅ Added person: **{name}** (`{new_id}`)"
    )

@hub_group.command(name="add_place", description="Add a place to the Knowledge Hub")
@app_commands.describe(
    name="Place name",
    modern_name="Modern name if different",
    variants="Alternative forms, comma-separated",
    region="Region (e.g. Bern, Luzern)",
    wikidata="Wikidata Q-ID",
)
async def hub_add_place(
    interaction: discord.Interaction,
    name: str,
    modern_name: str = "",
    variants: str = "",
    region: str = "",
    wikidata: str = "",
):
    places = orc.hub.get_places()
    new_id = f"hub_loc_{len(places)+1:04d}"
    place = {
        "id": new_id,
        "name": name,
        "modern_name": modern_name or name,
        "variants": [v.strip() for v in variants.split(",") if v.strip()],
        "region": region,
        "wikidata_id": wikidata or None,
        "gnd_id": None,
    }
    orc.hub_add_place(place)
    await interaction.response.send_message(
        f"✅ Added place: **{name}** (`{new_id}`)"
    )

tree.add_command(hub_group)


# ── /status ───────────────────────────────────────────────────────────────

@tree.command(name="status", description="System status overview")
async def cmd_status(interaction: discord.Interaction):
    transcriptions = list(config.TRANSCRIPTION_DIR.glob("*.txt"))
    descriptions   = list(config.DESCRIPTION_DIR.glob("*.md"))
    hot_files      = [
        p for p in config.HOT_FOLDER.iterdir()
        if p.is_file() and p.suffix.lower() in config.ALL_EXTENSIONS
    ]

    msg = (
        f"## 🏛️ Agentic Historian – System Status\n"
        f"**Transcriptions:** {len(transcriptions)}\n"
        f"**Descriptions:** {len(descriptions)}\n"
        f"**Hot folder queue:** {len(hot_files)} file(s)\n"
        f"**Model:** `{config.CLAUDE_MODEL}`\n\n"
        f"{orc.hub_summary()}"
    )
    await interaction.response.send_message(msg)


# ── /help ─────────────────────────────────────────────────────────────────

@tree.command(name="help", description="Show all commands")
async def cmd_help(interaction: discord.Interaction):
    msg = """## 🏛️ Agentic Historian – Commands

**Pipeline**
`/process [file]` – Full pipeline (A→B→C) on one document  
`/hot_folder` – Process everything in the hot folder  
`/corpus [name] [doc_ids]` – Corpus analysis (Agent D)  
`/report` – Meta-agent resource report (Agent E)  

**Individual Agents**
`/agent a [file]` – Agent A: Handwriting Text Recognition  
`/agent b [doc_id] [file?]` – Agent B: Source Description  
`/agent c [doc_id]` – Agent C: Entity Extraction & Linking  
`/agent d [corpus_name]` – Agent D: Corpus Analysis  
`/agent e` – Agent E: Meta-Report  

**Knowledge Hub**
`/hub list` – Show hub contents  
`/hub add_keyword [term]` – Add controlled vocab term  
`/hub add_type [type]` – Add document type  
`/hub add_person [name] ...` – Register a historical person  
`/hub add_place [name] ...` – Register a historical place  

**System**
`/status` – Overview of processed docs and queue  
`/help` – This message  
"""
    await interaction.response.send_message(msg)


# ── Bot events ─────────────────────────────────────────────────────────────

@client.event
async def on_ready():
    global orc
    orc = Orchestrator()
    logger.info(f"Logged in as {client.user}")

    guild = discord.Object(id=config.DISCORD_GUILD_ID) if config.DISCORD_GUILD_ID else None
    if guild:
        tree.copy_global_to(guild=guild)
        await tree.sync(guild=guild)
        logger.info(f"Commands synced to guild {config.DISCORD_GUILD_ID}")
    else:
        await tree.sync()
        logger.info("Commands synced globally (may take up to 1 hour)")

    logger.info("🏛️ Agentic Historian bot is ready!")


# ── Hot-folder watcher ────────────────────────────────────────────────────

async def hot_folder_watcher():
    """Background task: periodically check hot folder and auto-process new files."""
    if not config.ENABLE_HOT_FOLDER_WATCH:
        return
    logger.info("[HotFolder] Watcher started")
    while True:
        await asyncio.sleep(60)  # check every 60 seconds
        images = [
            p for p in config.HOT_FOLDER.iterdir()
            if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
        ]
        if images and orc:
            logger.info(f"[HotFolder] {len(images)} new file(s) detected – processing…")
            await orc.process_hot_folder()


@client.event
async def setup_hook():
    client.loop.create_task(hot_folder_watcher())


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not config.DISCORD_BOT_TOKEN:
        raise ValueError("DISCORD_BOT_TOKEN not set in .env")
    client.run(config.DISCORD_BOT_TOKEN)
