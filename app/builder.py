"""Deterministic catalogue validation and Unity-ready game construction."""
import json
import re
from pathlib import Path

from pydantic import ValidationError

from app.models import GameConfig, GameIntent, GameObject, Hints, Reward, WinCondition

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DURATION = 120
DEFAULT_DIFFICULTY = "medium"
DEFAULT_OBJECT_COUNT = 3
DEFAULT_SCORE = 20


class GameError(Exception):
    def __init__(self, status: str, message: str, details: list[str] | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.details = details or []

    def as_dict(self):
        return {"status": self.status, "message": self.message, "details": self.details}


def load_catalog(path: Path | None = None) -> dict:
    with open(path or ROOT / "data" / "catalog.json", encoding="utf-8") as handle:
        return json.load(handle)


def _phrase_in_prompt(phrase: str, prompt: str) -> bool:
    """Checks a literal, complete quoted span; semantic alignment is still an LLM task."""
    phrase = " ".join(phrase.casefold().split())
    text = " ".join(prompt.casefold().split())
    if len(phrase) < 3:
        return False
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def _explicit_duration_seconds(prompt: str) -> int | None:
    """Normalize explicit numeric time units; an unqualified 'under 3 minutes' -> 180."""
    minutes = [int(n) * 60 for n in re.findall(r"\b(\d+)\s*(?:minutes?|mins?)\b", prompt, re.I)]
    seconds = [int(n) for n in re.findall(r"\b(\d+)\s*(?:seconds?|secs?)\b", prompt, re.I)]
    values = minutes + seconds
    if len(set(values)) > 1:
        raise GameError("needs_clarification", "Several conflicting duration limits were requested")
    return values[0] if values else None


def build_game(intent_data: dict | GameIntent, prompt: str, catalog: dict) -> GameConfig:
    """Build or reject: no model-generated IDs or scores bypass these checks."""
    try:
        intent = GameIntent.model_validate(intent_data)
    except ValidationError as exc:
        raise GameError("invalid_intent", "LLM intent failed schema validation") from exc

    if intent.status == "unsupported" or intent.unsupported_elements:
        raise GameError("unsupported", "Request contains unsupported elements", intent.unsupported_elements)

    if intent.status == "needs_clarification":
        raise GameError("needs_clarification", "More information is required", intent.clarification_questions)

    missing = [field for field in ("game_template", "environment") if getattr(intent, field) is None]
    if missing:
        raise GameError("needs_clarification", "Specify a game template and environment", missing)

    template_ids = {item["id"] for item in catalog["game_templates"]}
    environment_ids = {item["id"] for item in catalog["environments"]}
    all_asset_ids = [item["id"] for item in catalog["assets"]]
    allowed_assets = set(all_asset_ids)
    rules = catalog["rules"]

    if intent.game_template not in template_ids:
        raise GameError("unsupported", f"Unsupported game template: {intent.game_template}")
    if intent.game_template == "quiz_hunt":
        raise GameError("unsupported", "Quiz Hunt needs a question bank and quiz schema not supplied in the assignment")
    if intent.environment not in environment_ids:
        raise GameError("unsupported", f"Unsupported environment: {intent.environment}")

    requested = []
    for item in intent.assets:
        if item.asset_id not in allowed_assets:
            raise GameError("unsupported", f"Asset not in catalogue: {item.asset_id}")
        if not _phrase_in_prompt(item.evidence, prompt):
            raise GameError("invalid_intent", f"No supporting quote in user request for {item.asset_id}")
        if item.asset_id in requested:
            raise GameError("invalid_intent", f"Duplicate requested asset: {item.asset_id}")
        requested.append(item.asset_id)

    count = intent.object_count
    if count is None:
        count = len(requested) if requested else DEFAULT_OBJECT_COUNT
    if not rules["hidden_objects"]["min"] <= count <= rules["hidden_objects"]["max"]:
        raise GameError("unsupported", f"Object count {count} is outside 1-10")
    if count > len(all_asset_ids):
        raise GameError("unsupported", f"Only {len(all_asset_ids)} unique catalogue assets exist; duplicate instances are not assumed")
    if len(requested) > count:
        raise GameError("unsupported", "More explicit assets than the requested object count")

    # Preserve explicit assets. Fill unspecified slots deterministically in catalogue order.
    selected = list(requested)
    selected.extend(asset for asset in all_asset_ids if asset not in selected)  # preserves catalogue order
    selected = selected[:count]

    # Assessment convention: '3 minutes' AND 'under 3 minutes' -> 180 seconds.
    explicit_duration = _explicit_duration_seconds(prompt)
    duration = (
        explicit_duration if explicit_duration is not None
        else (intent.duration_seconds if intent.duration_seconds is not None else DEFAULT_DURATION)
    )
    if not rules["duration_seconds"]["min"] <= duration <= rules["duration_seconds"]["max"]:
        raise GameError("unsupported", "Duration must be between 30 and 180 seconds")

    difficulty = intent.difficulty or DEFAULT_DIFFICULTY
    if difficulty not in rules["difficulty"]:
        raise GameError("unsupported", f"Unknown difficulty: {difficulty}")
    hints = intent.max_hints if intent.max_hints is not None else 0
    if not rules["hints"]["min"] <= hints <= rules["hints"]["max"]:
        raise GameError("unsupported", "Hints must be between 0 and 5")

    if intent.reward_type not in (None, "card"):
        raise GameError("unsupported", f"Unsupported reward type: {intent.reward_type}")
    if intent.reward_type and not intent.reward_requested:
        raise GameError("invalid_intent", "Reward type provided but no reward was requested")

    if intent.score_per_object is not None:
        score_limit = rules["score_per_object"]
        if not score_limit["min"] <= intent.score_per_object <= score_limit["max"]:
            raise GameError("unsupported", "Object score must be between 10 and 100")
    if intent.scoring_mode == "increasing" and intent.score_per_object is not None:
        raise GameError("needs_clarification", "Fixed score conflicts with increasing-score requirement")

    objects = []
    for index, asset_id in enumerate(selected):
        # Simple illustrative point tiers, not claims about asset placement difficulty.
        score = (
            min(100, 20 + 20 * (index // 2))
            if intent.scoring_mode == "increasing"
            else (intent.score_per_object if intent.score_per_object is not None else DEFAULT_SCORE)
        )
        objects.append(GameObject(asset_id=asset_id, score=score))

    reward = Reward(type="card", reward_id="demo_card_01") if intent.reward_requested else None
    try:
        game = GameConfig(
            game_template=intent.game_template,
            environment=intent.environment,
            duration_seconds=duration,
            difficulty=difficulty,
            objects=objects,
            hints=Hints(enabled=hints > 0, max_hints=hints),
            win_condition=WinCondition(type="find_all_objects"),
            reward=reward,
        )
    except ValidationError as exc:
        raise GameError("invalid_configuration", "Final game failed schema validation") from exc

    # The game schema validates types/ranges. This second check enforces dynamic catalogue IDs.
    if any(obj.asset_id not in allowed_assets for obj in game.objects):
        raise GameError("invalid_configuration", "Final game references an unknown asset")
    return game
