"""The only public endpoint required by the technical assignment."""
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.builder import GameError, build_game, load_catalog
from app.generator import GroqExtractor, ModelError
from app.models import GameConfig
from app.retrieval import Retriever


class GameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=2000)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Prompt cannot be blank")
        return value


class GameService:
    def __init__(self, retriever=None, extractor=None, catalog=None):
        self.catalog = catalog if catalog is not None else load_catalog()
        self.retriever = retriever if retriever is not None else Retriever()
        self.extractor = extractor if extractor is not None else GroqExtractor()

    def generate(self, prompt: str) -> GameConfig:
        evidence = self.retriever.category_aware(prompt)
        intent = self.extractor.extract(prompt, evidence, self.catalog)
        return build_game(intent, prompt, self.catalog)


@lru_cache(maxsize=1)
def get_service() -> GameService:
    # Lazy init: Swagger /health and offline API tests work without an API key.
    try:
        return GameService()
    except (RuntimeError, ImportError) as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "message": "Set GROQ_API_KEY and run: python -m scripts.index",
            },
        ) from exc


app = FastAPI(title="OTRAVERSE AI Game Generator POC", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-game", response_model=GameConfig, response_model_exclude_none=True)
def generate_game(request: GameRequest, service: GameService = Depends(get_service)):
    try:
        return service.generate(request.prompt)
    except GameError as exc:
        # Invalid catalogue input or missing details should never yield a fake game.
        status_code = 500 if exc.status == "invalid_configuration" else 422
        raise HTTPException(status_code=status_code, detail=exc.as_dict()) from exc
    except ModelError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"status": "model_error", "message": str(exc)},
        ) from exc
