"""Run the five required test cases through the live FastAPI endpoint."""
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    cases = json.loads((ROOT / "examples" / "demo_prompts.json").read_text(encoding="utf-8"))
    results = []
    for case in cases:
        try:
            response = httpx.post(
                "http://127.0.0.1:8000/generate-game", json={"prompt": case["prompt"]}, timeout=90
            )
            result = {
                "scenario": case["scenario"], "http_status": response.status_code,
                "expected_http_status": case["expected_http_status"],
                "passed_status_check": response.status_code == case["expected_http_status"],
                "response": response.json(),
            }
            if response.status_code == 200:
                game = result["response"]
                if case["scenario"] == "compatible":
                    result["passed_status_check"] &= (
                        game["duration_seconds"] == 180
                        and len(game["objects"]) == 5
                        and game["hints"]["max_hints"] == 3
                        and game.get("reward", {}).get("reward_id") == "demo_card_01"
                    )
                if case["scenario"] == "complex_multi_constraint":
                    expected = {"red_shoe_01", "book_02", "headphones_01", "key_01"}
                    result["passed_status_check"] &= (
                        game["duration_seconds"] == 150
                        and {x["asset_id"] for x in game["objects"]} == expected
                        and all(x["score"] == 30 for x in game["objects"])
                    )
        except httpx.HTTPError as exc:
            result = {"scenario": case["scenario"], "error": str(exc), "passed_status_check": False}
        results.append(result)
        print(f"{case['scenario']}: {'PASS' if result['passed_status_check'] else 'REVIEW'}")
    output = ROOT / "examples" / "demo_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
