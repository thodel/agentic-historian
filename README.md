# Agentic Historian – Discord Bot

Ein agentisches KI-System für die historische Quellenerschliessung, gesteuert über Discord.

## Architektur

```
Discord Bot (bot.py)
    │
    └── Orchestrator (orchestrator.py)
            ├── Agent A – Text Recognition (HTR via Gemini 3.1 Pro Vision)
            ├── Agent B – Source Description (Classification, Keywords, Visual)
            ├── Agent C – Entity Extraction & Linking (NER + Wikidata/GND)
            ├── Agent D – Corpus Analysis (Topics, Taxonomien, Care, Voyant)
            └── Agent E – Meta Agent (Resources, Costs, Improvement Suggestions)
                    │
                    └── Knowledge Hub (controlled vocab, persons, places, doc types)
```

## Setup

### 1. Voraussetzungen

- Python 3.11+
- Discord Bot Token ([discord.com/developers](https://discord.com/developers))
- Anthropic API Key ([console.anthropic.com](https://console.anthropic.com)) – für Textmodelle
- Gemini API Key (Google AI Studio) – für Vision/HTR
- Optional: HuggingFace Token, GitHub Token

### 2. Installation

```bash
git clone <repo>
cd agentic_historian

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Sprachmodelle für NER (optional, für Offline-NER)
python -m spacy download de_core_news_lg
```

### 3. Konfiguration

```bash
cp .env.example .env
# .env bearbeiten und Tokens eintragen
```

### 4. Bot starten

```bash
python bot.py
```

---

## Discord Commands

### Pipeline

| Befehl | Beschreibung |
|--------|-------------|
| `/process [file]` | Vollständige Pipeline (Agent A→B→C) auf ein Dokument |
| `/hot_folder` | Alle Dateien im Hot Folder verarbeiten |
| `/corpus [name]` | Korpusanalyse (Agent D) |
| `/report` | Meta-Agent Ressourcenbericht (Agent E) |

### Einzelne Agents

| Befehl | Agent | Funktion |
|--------|-------|---------|
| `/agent a [file]` | A | HTR – Handschriftenerkennung |
| `/agent b [doc_id]` | B | Quellenerschliessung & Metadaten |
| `/agent c [doc_id]` | C | Entitätsextraktion & -verlinkung |
| `/agent d` | D | Korpusanalyse + Voyant Tools Link |
| `/agent e` | E | Meta-Bericht |

### Knowledge Hub

```
/hub list                         – Übersicht
/hub add_keyword [term]           – Kontrolliertes Vokabular ergänzen
/hub add_type [type]              – Dokumenttyp hinzufügen
/hub add_person [name] [...]      – Person registrieren
/hub add_place [name] [...]       – Ort registrieren
```

---

## Workflow

### Einzelnes Dokument

1. Bild der Quelle (JPG/PNG/TIFF) an Discord anhängen
2. `/process [file]` ausführen
3. Der Bot gibt schrittweise Statusmeldungen aus
4. Am Ende: Zusammenfassung aller Agenten-Outputs

### Batch / Hot Folder

1. Dateien in `data/hot_folder/` ablegen
2. `/hot_folder` in Discord ausführen  
   **oder** automatisch: Der Bot prüft den Ordner alle 60 Sekunden

### Mehrseitige Dokumente

Dateien mit gleichem Stamm werden als ein Dokument behandelt:
- `missive_001_p1.jpg`, `missive_001_p2.jpg` → Dokument `missive_001`

---

## Knowledge Hub

Der Knowledge Hub wird von den Historiker:innen befüllt und steuert das Verhalten der Agents:

- **Dokumenttypen** → Agent B (Klassifikation)  
- **Kontrolliertes Vokabular** → Agent B (Keywords), Agent D (Taxonomieanalyse)  
- **Personen** → Agent C (Entity Linking)  
- **Orte** → Agent C (Entity Linking)  

Daten werden als JSON in `knowledge_hub/data/` gespeichert.

---

## Output-Struktur

```
data/
├── hot_folder/          # Eingang (neue Bilder hier ablegen)
│   └── processed/       # Nach Verarbeitung verschoben
├── transcriptions/      # Agent A: *.txt Transkriptionen
├── descriptions/        # Agent B: *.md Quellenerschliessung
├── outputs/
│   ├── *_entities.json  # Agent C: Entitäten
│   ├── *_entities.md    # Agent C: Entitäten (Markdown)
│   ├── corpus_*/        # Agent D: Korpusanalyse
│   │   ├── corpus.txt
│   │   ├── stats.json
│   │   ├── topics.json
│   │   ├── taxonomy.json
│   │   ├── care_analysis.json
│   │   └── report.md
│   └── meta_report.md   # Agent E: Meta-Bericht
└── corpus/              # Langzeitspeicher Korpus
```

---

## Anpassungen & Erweiterungen

### Eigenes HTR-Modell
In `agents/text_recognition.py` kann in `process_file()` alternativ ein Transkribus-API-Aufruf oder ein lokales OCR4all-Modell eingebunden werden. Das aktuelle System nutzt Gemini 3.1 Pro (Vision) als Fallback für alle Schrifttypen.

### HuggingFace Upload
In `storage/hf_storage.py` (erweiterbar) kann der Upload der Transkriptionen auf ein HuggingFace Dataset ergänzt werden.

### Knowledge Graph (CIDOC-CRM/SDHSS)
Die Entitäten aus Agent C können als CIDOC-CRM-konforme Linked Data exportiert werden. Dafür kann `knowledge_hub/hub.py` um einen RDF-Serializer (rdflib) erweitert werden.

---

## Projektbezug

Implementierung der Projektskizze **"Agentic Historian: Praxeologie in Methode und Material"**:
- Teilprojekt **Taxonomien des Sozialen**: Agent B (Soziale Taxonomie-Terms) + Agent D (Taxonomy-Analyse)
- Teilprojekt **Praxis und Preis der Care**: Agent B (Care-Flag) + Agent C (Care Actors) + Agent D (Care-Analyse)
- **Prüfbarkeit**: Alle Outputs sind nachvollziehbar als Markdown/JSON gespeichert
- **Historiker:in im Loop**: Discord-Interface ermöglicht gezielte manuelle Eingriffe und Hub-Pflege
