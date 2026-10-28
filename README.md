# OTRAVERSE — AI Game Generator POC

This project is a proof of concept for the OTRAVERSE AI/RAG Engineering assignment. It takes a game description written in natural language and returns a validated JSON configuration using a limited catalogue of game templates, environments, assets, and rules.

The goal is to demonstrate a reliable generation workflow, not to build the Unity game itself. A language model interprets the request, but the final configuration is assembled and checked in Python so that unsupported mechanics and invalid values are not silently introduced.

## 1. Architecture

```text
User request
     |
     v
POST /generate-game  (FastAPI)
     |
     v
Catalogue retrieval  (Sentence Transformers + ChromaDB)
     |
     v
Intent extraction  (Groq LLM + structured JSON schema)
     |
     v
Game builder  (Python rules and catalogue checks)
     |
     v
Final validation  (Pydantic)
     |
     +---- Valid request ----------> Unity-ready JSON (HTTP 200)
     |
     +---- Unsupported / unclear --> Explanation (HTTP 422)
```

`data/catalog.json` is the source of truth. It contains the three templates, three environments, seven assets, and five rule definitions provided in the assignment. The RAG layer retrieves relevant catalogue entries; the LLM extracts the user's intended game settings; and the builder applies defaults and validates the result. Retrieved assets are treated as candidates, not as assets the user necessarily requested.

The LLM outputs an intermediate `GameIntent`, while the API returns a separate `GameConfig` schema. Keeping these two stages apart makes the validation rules explicit and prevents the model from directly inventing Unity-facing fields.

### Project structure

```text
app/
  main.py          FastAPI endpoints and generation pipeline
  retrieval.py     ChromaDB indexing and semantic retrieval
  generator.py     Groq structured intent extraction
  models.py        Pydantic intent and game schemas
  builder.py       Game construction and catalogue validation

data/
  catalog.json          Supported content and limits
  retrieval_eval.json   Labeled retrieval test queries

scripts/
  index.py         Create the local vector index
  evaluate.py      Evaluate retrieval performance
  demos.py         Run the five assignment scenarios

tests/
  test_project.py  Offline unit and API tests

examples/
  demo_prompts.json
  sample_game_config.json

requirements.txt
.env.example
.gitignore
README.md
```

## 2. Setup and usage

Python 3.11 is recommended. The following commands are for Windows PowerShell and should be run from the project root.

**Install dependencies**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**Configure the Groq API key**

```powershell
Copy-Item .env.example .env
```

Edit `.env` and enter a valid key:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

The `.env` file is excluded from Git. Do not commit API keys. Any key previously shared publicly should be revoked and replaced.

**Build the knowledge-base index**

```powershell
python -m scripts.index
```

This embeds the catalogue entries and stores the vectors locally in `data/chroma_db/`. The embedding model may be downloaded on the first run. Rebuild the index after modifying the catalogue.

**Start the API**

```powershell
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the API through Swagger UI. The main endpoint is `POST /generate-game` and accepts a JSON body containing a `prompt` string.

```json
{
  "prompt": "Create a game in a messy bedroom. The player must find 5 objects in less than 3 minutes. Harder objects should give more points. Add 3 hints and a reward at the end."
}
```

A successful response contains the game template, environment, duration, difficulty, object IDs and scores, hints, win condition, and optional reward. Each object has only `asset_id` and `score`, matching the structure requested in the assignment. The full sample is in `examples/sample_game_config.json`.

The original request asks for **five** objects, so the application returns five. The assignment's sample JSON lists only three objects; I have followed the explicit object count while keeping the same JSON structure. For this POC, a three-minute limit (including the sample wording “less than 3 minutes”) is represented as **180 seconds**, following the assignment example. This is a documented inclusive-cap convention rather than a literal strict inequality.

## 3. Technical choices

**Retrieval.** I used `sentence-transformers/all-MiniLM-L6-v2` to embed 18 short catalogue records and ChromaDB to store them with cosine-distance search. Retrieval is category-aware so that templates, environments, assets, and rules are considered separately. The model also receives the complete list of allowed IDs and rules, which is practical for a catalogue this small and helps with requests where top-K retrieval misses a named asset. Every generated ID is checked against the catalogue regardless of the retrieval score.

**LLM and structured output.** Groq serves `openai/gpt-oss-20b`, configured to return a structured `GameIntent` using JSON Schema. The model's responsibility is to identify the request, including missing or unsupported details. Named assets must be supported by quotations from the original prompt; the builder checks those quotations rather than accepting every asset suggested by retrieval.

**Game construction and validation.** Python selects assets from the controlled list and applies business rules. Pydantic validates the final response shape, while the builder checks catalogue IDs, distinct assets, object count, duration, hint limits, scores, and reward handling. Invalid or incompatible requests are rejected rather than adjusted silently.

**API.** FastAPI exposes `POST /generate-game` and `GET /health`. Valid configurations return HTTP 200. Unsupported or ambiguous requests return HTTP 422 with an explanation. Provider failures are reported separately, rather than returning a misleading game configuration.

### Assumptions used in the POC

The assignment defines valid ranges but not every default or game-specific policy. I therefore used a small number of deterministic choices:

| Case | Policy |
| --- | --- |
| Duration unspecified | 120 seconds |
| Difficulty unspecified | `medium` |
| Object count unspecified | Number of named assets, or 3 when none are named |
| Fewer named assets than the requested count | Fill the remaining slots with allowed assets in catalogue order |
| Hints unspecified | 0 |
| Score unspecified | 20 points per object |
| Harder objects should earn more | Illustrative score tiers: 20, 20, 40, 40, 60, etc. |
| Reward requested | Use the example card reward `demo_card_01` |

These choices are reproducible but are **not** claims about real object placement or hiding difficulty in Unity.

## 4. Tests and evaluation

Run the offline unit and API tests:

```powershell
python -m pytest -q
```

They cover schema validation, rule boundaries, asset references, incompatible requirements, error handling, and API behavior without making live Groq calls.

To evaluate retrieval after indexing:

```powershell
python -m scripts.evaluate
```

The evaluation compares full-prompt retrieval with category-aware retrieval using the same five-document budget. It reports Precision@5, Recall@5, MRR@5, and complete-query coverage for 24 labeled supported queries. Six additional unsupported requests are checked separately because retrieval relevance alone does not establish whether a game is supported. Results are written to `examples/retrieval_report.json`.

To run the five required demonstration scenarios, keep FastAPI running in one terminal and execute this in another:

```powershell
python -m scripts.demos
```

The scenarios cover a compatible request, an ambiguous request, a missing asset, an unsupported game mechanic, and a request with multiple constraints. The script writes the responses to `examples/demo_results.json`. Since these tests use a live LLM, the returned configurations and rejection reasons should also be reviewed manually; matching the expected HTTP status is not sufficient on its own.

## 5. Current limitations

This is deliberately a small POC and has a few important boundaries:

- **Game mechanics:** Hidden Objects is supported. Treasure Hunt uses a simplified collect-all-objects flow. Quiz Hunt is searchable but cannot be generated as a playable quiz without question content and a quiz-specific schema.
- **Unity integration:** The JSON identifies game elements and rules but does not include object placement, asset-bundle references, animations, or scene behavior. It has not been executed in Unity.
- **Asset count:** Although the numerical rule allows up to ten hidden objects, the catalogue contains seven distinct assets. The implementation rejects requests for more than seven unique objects rather than inventing duplicate instances.
- **Scoring:** Increasing scores are an illustrative policy. Actual object difficulty depends on gameplay and placement information that the current catalogue does not provide.
- **Semantic accuracy:** The LLM can still misunderstand unusual requests or unsupported mechanics. Pydantic and catalogue checks protect the output structure and known limits, but broader end-to-end testing is needed.
- **Evaluation scale:** Eighteen catalogue entries and a small labeled test set demonstrate the approach, not production-scale retrieval performance.

## 6. How I would take it toward production

I would keep the same separation between retrieval, intent extraction, and deterministic validation, then improve the parts needed for a real Unity workflow.

**First, I would agree on a versioned JSON contract with the Unity team.** Each game type needs a clear definition of its required fields. Asset placement, valid scene locations, and supported interactions should come from Unity-side data rather than being guessed by the LLM. Quiz Hunt would need a question bank before it could be enabled.

**Next, I would make the catalogue easier to maintain.** The static JSON could become a versioned content store with explicit links between templates, environments, assets, and supported mechanics. Changes would be validated and reindexed as part of the content update process.

**I would then improve evaluation using unseen requests and real game configurations.** Retrieval metrics would remain useful, but the main measure would be whether the final JSON preserves the user's constraints and can actually be loaded by Unity. Integration tests would reveal problems that schema validation alone cannot catch.

**Finally, I would prepare the API for real traffic.** That means authentication, sensible rate limits, request timeouts, monitoring, cost tracking, and safe handling of prompts and API keys. I would introduce caching or additional infrastructure only where usage and performance justify it.

For this assignment, the boundary is intentional: the project demonstrates how to turn a user's request into a controlled, validated configuration, while leaving actual Unity gameplay and production infrastructure as the next stage.
