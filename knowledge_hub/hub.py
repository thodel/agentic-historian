"""
knowledge_hub/hub.py – Knowledge Hub
Provides controlled vocabularies, entity registries and document type definitions.
Historians populate this hub via Discord commands or direct YAML/JSON editing.
"""
import json
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

import config

HUB_DIR = config.KNOWLEDGE_HUB_DIR

DEFAULTS = {
    "document_types.json": [
        "Missive", "Ratsprotokoll", "Urteilsbrief", "Bürgschaftsbrief",
        "Schuldbrief", "Kaufbrief", "Steuerregister", "Verhörprotokoll",
        "Mandate", "Rechnung", "Inventar", "Testament", "Pfandbrief",
        "Instruktion", "Supplikation", "Satzung / Ordnung",
    ],
    "controlled_vocabulary.json": [
        # Social taxonomy terms (Taxonomien des Sozialen)
        "arme lüt", "erbar lüt", "Bürger", "Burger", "Hintersässe",
        "Juden", "Zigeuner", "Vaganten", "Söldner", "Dienstbot",
        "Knecht", "Magd", "Junckfrow", "Witwe", "Waise",
        "gesellen", "meister", "lehrling",
        # Care terms (Praxis und Preis der Care)
        "versorgung", "pflege", "dienst", "erziehung", "hut",
        "spital", "almosen", "fürsorge", "lohn", "pfand",
        # Administrative terms
        "vogt", "schultheiss", "rat", "amtmann", "richter",
        "steuer", "zins", "schuld", "pfand", "erbe", "gut",
        # Conflict / social order
        "friede", "fehde", "klage", "urteil", "strafe", "buss",
        "ehre", "unehrlich", "scham", "treue",
    ],
    "language_varieties.json": [
        "Alemannic German (15th c.)",
        "Bernese Middle German",
        "Lucerne Middle German",
        "Medieval Latin",
        "Mixed German-Latin",
        "Old French",
    ],
    "persons.json": [],   # populated by historians
    "places.json": [],    # populated by historians
    "organisations.json": [],
}


class KnowledgeHub:
    """Manages all knowledge hub data files."""

    def __init__(self):
        HUB_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_defaults()

    def _ensure_defaults(self):
        for filename, default in DEFAULTS.items():
            path = HUB_DIR / filename
            if not path.exists():
                path.write_text(
                    json.dumps(default, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

    def _load(self, filename: str) -> list | dict:
        path = HUB_DIR / filename
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return []

    def _save(self, filename: str, data: list | dict):
        path = HUB_DIR / filename
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── Document types ─────────────────────────────────────────────────────
    def get_document_types(self) -> list[str]:
        return self._load("document_types.json")

    def add_document_type(self, dtype: str):
        types = self.get_document_types()
        if dtype not in types:
            types.append(dtype)
            self._save("document_types.json", types)
            logger.info(f"[Hub] Added document type: {dtype}")

    # ── Controlled vocabulary ──────────────────────────────────────────────
    def get_controlled_vocabulary(self) -> list[str]:
        return self._load("controlled_vocabulary.json")

    def add_keyword(self, keyword: str):
        vocab = self.get_controlled_vocabulary()
        if keyword not in vocab:
            vocab.append(keyword)
            self._save("controlled_vocabulary.json", vocab)

    # ── Persons ────────────────────────────────────────────────────────────
    def get_persons(self) -> list[dict]:
        return self._load("persons.json")

    def add_person(self, person: dict):
        """
        person = {
          "id": "hub_p_001",
          "name": "Heinrich von Wiler",
          "variants": ["Hainricus de Villa", "H. Wiler"],
          "role": "Vogt",
          "active_period": "1430–1450",
          "location": "Thun",
          "gnd_id": null,
          "wikidata_id": null,
          "notes": ""
        }
        """
        persons = self.get_persons()
        # Update if exists
        for i, p in enumerate(persons):
            if p.get("id") == person.get("id"):
                persons[i] = person
                self._save("persons.json", persons)
                return
        persons.append(person)
        self._save("persons.json", persons)
        logger.info(f"[Hub] Added person: {person.get('name')}")

    def find_person(self, name: str) -> Optional[dict]:
        name_lower = name.lower()
        for p in self.get_persons():
            if name_lower == p.get("name", "").lower():
                return p
            for v in p.get("variants", []):
                if name_lower == v.lower():
                    return p
        return None

    # ── Places ─────────────────────────────────────────────────────────────
    def get_places(self) -> list[dict]:
        return self._load("places.json")

    def add_place(self, place: dict):
        """
        place = {
          "id": "hub_loc_001",
          "name": "Thun",
          "variants": ["Tun", "Thunum"],
          "modern_name": "Thun",
          "region": "Bern",
          "gnd_id": null,
          "wikidata_id": "Q45367",
          "coordinates": {"lat": 46.758, "lon": 7.628}
        }
        """
        places = self.get_places()
        for i, pl in enumerate(places):
            if pl.get("id") == place.get("id"):
                places[i] = place
                self._save("places.json", places)
                return
        places.append(place)
        self._save("places.json", places)

    def find_place(self, name: str) -> Optional[dict]:
        name_lower = name.lower()
        for pl in self.get_places():
            if name_lower == pl.get("name", "").lower():
                return pl
            for v in pl.get("variants", []):
                if name_lower == v.lower():
                    return pl
        return None

    # ── Organisations ──────────────────────────────────────────────────────
    def get_organisations(self) -> list[dict]:
        return self._load("organisations.json")

    def add_organisation(self, org: dict):
        orgs = self.get_organisations()
        for i, o in enumerate(orgs):
            if o.get("id") == org.get("id"):
                orgs[i] = org
                self._save("organisations.json", orgs)
                return
        orgs.append(org)
        self._save("organisations.json", orgs)

    # ── Summary ────────────────────────────────────────────────────────────
    def summary(self) -> str:
        return (
            f"📚 **Knowledge Hub Summary**\n"
            f"• Document types: {len(self.get_document_types())}\n"
            f"• Controlled vocabulary: {len(self.get_controlled_vocabulary())} terms\n"
            f"• Persons registered: {len(self.get_persons())}\n"
            f"• Places registered: {len(self.get_places())}\n"
            f"• Organisations: {len(self.get_organisations())}"
        )
