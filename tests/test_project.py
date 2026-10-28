"""Offline tests; no Groq key or model downloads required."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.builder import GameError, build_game, load_catalog
from app.generator import GroqExtractor, ModelError
from app.main import GameService, app, get_service
from app.models import GameConfig, GameIntent
from app.retrieval import catalogue_documents
from scripts.evaluate import metrics

ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog()
BASE_PROMPT = "Create a hidden objects game in a bedroom with a book"


def intent(**changes):
    base = {
        "status": "supported",
        "game_template": "hidden_objects",
        "environment": "bedroom",
        "assets": [],
        "object_count": 3,
        "duration_seconds": 180,
        "difficulty": "medium",
        "max_hints": 3,
        "score_per_object": None,
        "scoring_mode": None,
        "reward_requested": True,
        "reward_type": None,
        "unsupported_elements": [],
        "clarification_questions": [],
    }
    base.update(changes)
    return base


def build(data=None, prompt=BASE_PROMPT):
    return build_game(intent(**(data or {})), prompt, CATALOG)


def error_status(data, prompt=BASE_PROMPT):
    with pytest.raises(GameError) as exc:
        build(data, prompt)
    return exc.value.status


def test_catalogue_exact_contents():
    assert len(CATALOG["game_templates"]) == 3
    assert len(CATALOG["environments"]) == 3
    assert len(CATALOG["assets"]) == 7
    assert len(CATALOG["rules"]) == 5


def test_catalogue_18_documents_and_metadata():
    docs = catalogue_documents(CATALOG)
    assert len(docs) == 18
    assert len({d["id"] for d in docs}) == 18
    assert sum(d["metadata"]["category"] == "rule" for d in docs) == 5


def test_groq_schema_all_required_and_forbids_extra():
    schema = GameIntent.model_json_schema()
    assert set(schema["properties"]) == set(schema["required"])
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["AssetMention"]["additionalProperties"] is False


def test_nullable_bool_does_not_break_schema():
    result = GameIntent.model_validate(intent(reward_requested=None))
    assert result.reward_requested is False


def test_final_schema_has_no_object_difficulty():
    game = build()
    data = game.model_dump(exclude_none=True)
    assert set(data["objects"][0]) == {"asset_id", "score"}
    assert "difficulty" not in data["objects"][0]


def test_no_unrecognized_properties_in_final_game():
    payload = build().model_dump(exclude_none=True)
    payload["invented_mechanic"] = True
    with pytest.raises(ValidationError):
        GameConfig.model_validate(payload)


def test_3_minutes_is_180():
    assert build({"duration_seconds": 180}).duration_seconds == 180


def test_less_than_three_minutes_normalized_even_if_llm_returns_179():
    prompt = "Find 5 objects in less than 3 minutes in a bedroom"
    assert build({"duration_seconds": 179, "object_count": 5}, prompt).duration_seconds == 180


def test_two_conflicting_time_limits_are_rejected():
    prompt = "Find 5 objects in 2 minutes or 3 minutes in a bedroom"
    assert error_status({"duration_seconds": 180}, prompt) == "needs_clarification"


def test_explicit_seconds_override_inaccurate_llm_estimate():
    assert build({"duration_seconds": 179}, "Find objects for 150 seconds").duration_seconds == 150


def test_original_assignment_sample_uses_180_and_five_assets():
    game = build({"assets": [], "object_count": 5, "scoring_mode": "increasing"})
    output = game.model_dump(exclude_none=True)
    assert output["duration_seconds"] == 180
    assert len(output["objects"]) == 5
    assert [obj["score"] for obj in output["objects"]] == [20, 20, 40, 40, 60]
    assert output["reward"]["reward_id"] == "demo_card_01"
    assert output["hints"]["max_hints"] == 3
    assert output["win_condition"]["type"] == "find_all_objects"


def test_all_selected_assets_from_catalogue():
    game = build({"object_count": 5})
    allowed = {a["id"] for a in CATALOG["assets"]}
    assert {o.asset_id for o in game.objects}.issubset(allowed)


def test_mentioned_assets_preserved_without_extra_when_count_unspecified():
    text = "Find red shoe, book, headphones and key in a bedroom"
    game = build({
        "assets": [
            {"asset_id": "red_shoe_01", "evidence": "red shoe"},
            {"asset_id": "book_02", "evidence": "book"},
            {"asset_id": "headphones_01", "evidence": "headphones"},
            {"asset_id": "key_01", "evidence": "key"},
        ],
        "object_count": None,
    }, text)
    assert [o.asset_id for o in game.objects] == [
        "red_shoe_01", "book_02", "headphones_01", "key_01"
    ]


def test_unrequested_asset_rejected_without_evidence():
    assert error_status({"assets": [{"asset_id": "smartphone_01", "evidence": "smartphone"}]}) == "invalid_intent"


def test_quote_must_match_a_complete_word():
    assert error_status({"assets": [{"asset_id": "key_01", "evidence": "key"}]}, "Find a keyboard") == "invalid_intent"


def test_unsupported_asset_id():
    assert error_status({"assets": [{"asset_id": "bicycle_01", "evidence": "bicycle"}]}, "Find a bicycle") == "unsupported"


def test_no_duplicate_requested_assets():
    assert error_status({"assets": [
        {"asset_id": "book_02", "evidence": "book"},
        {"asset_id": "book_02", "evidence": "book"},
    ]}) == "invalid_intent"


def test_more_named_assets_than_count():
    assert error_status({"object_count": 1, "assets": [
        {"asset_id": "book_02", "evidence": "book"},
        {"asset_id": "red_shoe_01", "evidence": "red shoe"},
    ]}, "Find a book and red shoe") == "unsupported"


@pytest.mark.parametrize("n", [0, 8, 10, 11])
def test_object_count_outside_executable_catalogue(n):
    assert error_status({"object_count": n}) == "unsupported"


@pytest.mark.parametrize("duration", [0, 29, 181, 500])
def test_duration_bounds(duration):
    assert error_status({"duration_seconds": duration}) == "unsupported"


@pytest.mark.parametrize("hints", [-1, 6, 10])
def test_hints_bounds(hints):
    assert error_status({"max_hints": hints}) == "unsupported"


@pytest.mark.parametrize("score", [0, 9, 101])
def test_score_bounds(score):
    assert error_status({"score_per_object": score}) == "unsupported"


def test_missing_template_clarification():
    assert error_status({"game_template": None}) == "needs_clarification"


def test_missing_environment_clarification():
    assert error_status({"environment": None}) == "needs_clarification"


def test_model_unsupported_override():
    assert error_status({"status": "unsupported", "unsupported_elements": ["Formula 1 racing"]}) == "unsupported"


def test_unsupported_reward():
    assert error_status({"reward_type": "cash"}) == "unsupported"


def test_conflicting_scoring():
    assert error_status({"scoring_mode": "increasing", "score_per_object": 30}) == "needs_clarification"


def test_unspecified_reward_omitted():
    data = build({"reward_requested": False}).model_dump(exclude_none=True)
    assert "reward" not in data


def test_quiz_hunt_rejected_with_specific_reason():
    with pytest.raises(GameError, match="question bank"):
        build({"game_template": "quiz_hunt"})


def test_treasure_hunt_basic_collection_poc():
    game = build({"game_template": "treasure_hunt"})
    assert game.game_template == "treasure_hunt"
    assert game.win_condition.type == "find_all_objects"


def test_retrieval_precision_and_recall():
    score = metrics(["environment:bedroom", "asset:book_02", "asset:key_01"], ["environment:bedroom", "asset:book_02"])
    assert score["precision_at_5"] == 2 / 5
    assert score["recall_at_5"] == 1.0
    assert score["mrr_at_5"] == 1.0
    assert score["complete_coverage_at_5"] == 1.0


def test_retrieval_empty_labels_not_scored():
    with pytest.raises(ValueError):
        metrics([], [])


def test_all_five_demo_categories_present():
    cases = json.loads((ROOT / "examples" / "demo_prompts.json").read_text(encoding="utf-8"))
    assert {case["scenario"] for case in cases} == {
        "compatible", "ambiguous", "missing_asset", "unsupported_mechanic", "complex_multi_constraint"
    }


class FakeRetriever:
    def category_aware(self, prompt):
        return [{"id": "environment:bedroom", "text": "Bedroom", "metadata": {"category": "environment"}}]


class FakeExtractor:
    def __init__(self, output=None):
        self.output = output or intent()

    def extract(self, prompt, retrieved, catalog):
        assert retrieved  # The service really passes retrieved context to Groq.
        return GameIntent.model_validate(self.output)


def fake_service(data=None):
    return GameService(retriever=FakeRetriever(), extractor=FakeExtractor(data), catalog=CATALOG)


def test_pipeline_with_fakes_calls_retrieval_extraction_builder():
    game = fake_service().generate(BASE_PROMPT)
    assert isinstance(game, GameConfig)


def test_groq_extractor_strict_schema_mocked():
    sample = intent(assets=[{"asset_id": "book_02", "evidence": "book"}])
    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["response_format"]["json_schema"]["strict"] is True
            assert kwargs["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
            assert kwargs["messages"][1]["role"] == "user"
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(sample)))])
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    result = GroqExtractor(client=fake_client).extract(BASE_PROMPT, FakeRetriever().category_aware(BASE_PROMPT), CATALOG)
    assert result.assets[0].asset_id == "book_02"


def test_api_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200


def test_api_valid_request_exact_keys():
    app.dependency_overrides[get_service] = lambda: fake_service()
    try:
        response = TestClient(app).post("/generate-game", json={"prompt": BASE_PROMPT})
        assert response.status_code == 200, response.text
        body = response.json()
        assert set(body) == {"game_template", "environment", "duration_seconds", "difficulty", "objects", "hints", "win_condition", "reward"}
        assert set(body["objects"][0]) == {"asset_id", "score"}
    finally:
        app.dependency_overrides.clear()


def test_api_rejects_unsupported_mechanic():
    app.dependency_overrides[get_service] = lambda: fake_service(intent(status="unsupported", unsupported_elements=["Formula 1 racing"]))
    try:
        response = TestClient(app).post("/generate-game", json={"prompt": "Formula 1 racing in Tokyo"})
        assert response.status_code == 422
        assert response.json()["detail"]["status"] == "unsupported"
    finally:
        app.dependency_overrides.clear()


def test_api_needs_clarification():
    app.dependency_overrides[get_service] = lambda: fake_service(intent(game_template=None, status="needs_clarification"))
    try:
        response = TestClient(app).post("/generate-game", json={"prompt": "Make me a game"})
        assert response.status_code == 422
        assert response.json()["detail"]["status"] == "needs_clarification"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("payload", [{}, {"prompt": " "}, {"prompt": 1}, {"prompt": "hi", "foo": "bar"}])
def test_api_rejects_invalid_request(payload):
    app.dependency_overrides[get_service] = lambda: fake_service()
    try:
        response = TestClient(app).post("/generate-game", json=payload)
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_api_model_error_returns_503_without_private_details():
    class FailedService:
        def generate(self, prompt):
            raise ModelError("Groq rate limit reached", status_code=503)
    app.dependency_overrides[get_service] = lambda: FailedService()
    try:
        response = TestClient(app).post("/generate-game", json={"prompt": "Generate a game"})
        assert response.status_code == 503
        assert "GROQ_API_KEY" not in response.text
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("scenario", [
    "compatible", "ambiguous", "missing_asset", "unsupported_mechanic", "complex_multi_constraint"
])
def test_required_demo_scenarios_offline_end_to_end(scenario):
    cases = json.loads((ROOT / "examples" / "demo_prompts.json").read_text(encoding="utf-8"))
    case = next(case for case in cases if case["scenario"] == scenario)
    if scenario == "compatible":
        output = intent(assets=[], object_count=5, duration_seconds=180, scoring_mode="increasing")
    elif scenario == "ambiguous":
        output = intent(status="needs_clarification", game_template=None, environment=None,
                        clarification_questions=["Which game and environment?"])
    elif scenario == "missing_asset":
        output = intent(status="unsupported", unsupported_elements=["golden bicycle asset"])
    elif scenario == "unsupported_mechanic":
        output = intent(status="unsupported", unsupported_elements=["Formula 1 racing", "Tokyo"])
    else:
        output = intent(game_template="hidden_objects", environment="park", difficulty="hard",
                        object_count=4, duration_seconds=150, max_hints=2, score_per_object=30,
                        scoring_mode="fixed", reward_requested=True,
                        assets=[
                            {"asset_id": "red_shoe_01", "evidence": "red shoe"},
                            {"asset_id": "book_02", "evidence": "book"},
                            {"asset_id": "headphones_01", "evidence": "headphones"},
                            {"asset_id": "key_01", "evidence": "key"},
                        ])
    app.dependency_overrides[get_service] = lambda: fake_service(output)
    try:
        response = TestClient(app).post("/generate-game", json={"prompt": case["prompt"]})
        assert response.status_code == case["expected_http_status"], response.text
        if scenario == "compatible":
            game = response.json()
            assert len(game["objects"]) == 5
            assert game["duration_seconds"] == 180
            assert [x["score"] for x in game["objects"]] == [20, 20, 40, 40, 60]
        if scenario == "complex_multi_constraint":
            game = response.json()
            assert game["duration_seconds"] == 150
            assert len(game["objects"]) == 4
            assert all(x["score"] == 30 for x in game["objects"])
            assert game["hints"]["max_hints"] == 2
    finally:
        app.dependency_overrides.clear()
