import json
import re
from typing import Any

from .llm_router import analyze_complexity, classify_task, route_request
from .prompt_course_data import COURSE_CURRICULUM

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "you",
    "your",
}


def _build_learning_modules() -> list[dict[str, Any]]:
    modules: list[dict[str, Any]] = []
    for level in COURSE_CURRICULUM["levels"]:
        for module in level["modules"]:
            first_lesson = module["lessons"][0] if module["lessons"] else {}
            modules.append(
                {
                    "id": module["id"],
                    "level": level["level"],
                    "title": module["title"],
                    "focus": module["objective"],
                    "outcome": module["checkpoint_project"],
                    "practice_prompt": first_lesson.get("hands_on", ""),
                    "reflection_questions": module.get("review_questions", []),
                }
            )
    return modules


LEARNING_MODULES = _build_learning_modules()


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def _clip(value: float, min_value: float = 0.0, max_value: float = 10.0) -> float:
    return max(min_value, min(max_value, value))


def _has_any(text: str, patterns: list[str]) -> bool:
    lower = text.lower()
    return any(p in lower for p in patterns)


def get_learning_modules() -> list[dict[str, Any]]:
    return LEARNING_MODULES


def get_full_course_curriculum() -> dict[str, Any]:
    return COURSE_CURRICULUM


def get_all_lesson_ids() -> list[str]:
    lesson_ids: list[str] = []
    for level in COURSE_CURRICULUM["levels"]:
        for module in level["modules"]:
            for lesson in module["lessons"]:
                lesson_ids.append(lesson["id"])
    return lesson_ids


def normalize_completed_lessons(lessons: list[str]) -> list[str]:
    allowed = set(get_all_lesson_ids())
    unique = {lesson_id for lesson_id in lessons if lesson_id in allowed}
    return sorted(unique)


def progress_from_json(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return normalize_completed_lessons([str(item) for item in value])


def progress_to_json(completed_lessons: list[str]) -> str:
    return json.dumps(normalize_completed_lessons(completed_lessons))


def calculate_completion(completed_lessons: list[str]) -> tuple[int, int, float]:
    normalized = normalize_completed_lessons(completed_lessons)
    total_lessons = len(get_all_lesson_ids())
    completed_count = len(normalized)
    completion_pct = round((completed_count / total_lessons) * 100, 1) if total_lessons else 0.0
    return total_lessons, completed_count, completion_pct


def analyze_prompt_quality(prompt: str, goal: str | None = None) -> dict[str, Any]:
    lower = prompt.lower()
    words = prompt.split()
    word_count = len(words)
    sentence_count = max(1, len(re.findall(r"[.!?]+", prompt)))

    has_role = _has_any(lower, ["you are", "act as", "as a"])
    has_constraints = _has_any(
        lower,
        [
            "must",
            "should",
            "do not",
            "avoid",
            "at most",
            "limit",
            "json",
            "table",
            "bullet",
            "step",
            "format",
            "tone",
        ],
    )
    has_context = bool(goal) or _has_any(lower, ["context", "audience", "for ", "background"])
    has_eval = _has_any(lower, ["score", "evaluate", "rubric", "checklist", "verify", "success criteria"])
    has_structure = bool(re.search(r"(\d+\.)|(- )|(\n)", prompt))
    has_examples = _has_any(lower, ["example", "input", "output", "few-shot"])

    clarity = 5.0
    if 10 <= word_count <= 100:
        clarity += 2.0
    elif word_count < 10:
        clarity -= 1.5
    if sentence_count <= 4:
        clarity += 1.0
    if has_structure:
        clarity += 1.0

    context = 3.0 + (2.5 if has_role else 0) + (3.5 if has_context else 0)
    constraints = 2.5 + (4.5 if has_constraints else 0) + (1.0 if has_structure else 0)
    evaluation = 2.0 + (5.0 if has_eval else 0) + (1.0 if has_examples else 0)
    structure = 3.0 + (4.0 if has_structure else 0) + (1.5 if has_examples else 0)

    scores = {
        "clarity": round(_clip(clarity), 1),
        "context": round(_clip(context), 1),
        "constraints": round(_clip(constraints), 1),
        "evaluation": round(_clip(evaluation), 1),
        "structure": round(_clip(structure), 1),
    }
    overall = round(sum(scores.values()) / len(scores), 1)

    strengths: list[str] = []
    improvements: list[str] = []
    checklist: list[str] = []

    if has_role:
        strengths.append("Defines assistant role clearly.")
    else:
        improvements.append("Add an explicit role so the model adopts the right expertise.")

    if has_context:
        strengths.append("Includes context or audience signal.")
    else:
        improvements.append("Add background context and intended audience.")

    if has_constraints:
        strengths.append("Uses constraints or formatting guidance.")
    else:
        improvements.append("Add constraints (length, tone, must-have details, and format).")

    if has_eval:
        strengths.append("Includes evaluation language (rubric/checklist/verification).")
    else:
        improvements.append("Add success criteria to make output quality testable.")

    if has_structure:
        strengths.append("Prompt has structure that improves reliability.")
    else:
        improvements.append("Use numbered steps or sections to improve consistency.")

    checklist.extend(
        [
            "State role and objective in the first sentence.",
            "Provide context, audience, and domain assumptions.",
            "Specify hard constraints (length, tone, format, and exclusions).",
            "Request a self-check against a rubric before final answer.",
        ]
    )

    task_type = classify_task(prompt)
    complexity = analyze_complexity(prompt)
    rewritten_prompt = _rewrite_prompt(prompt, task_type, goal)

    return {
        "scores": scores,
        "overall_score": overall,
        "detected_task_type": task_type,
        "detected_complexity": complexity,
        "strengths": strengths,
        "improvements": improvements,
        "rewritten_prompt": rewritten_prompt,
        "checklist": checklist,
    }


def _rewrite_prompt(prompt: str, task_type: str, goal: str | None = None) -> str:
    role_map = {
        "coding": "a senior software engineer and technical mentor",
        "math": "an expert mathematician and teacher",
        "creative": "a creative writing coach",
        "factual": "a concise domain specialist",
    }
    role = role_map.get(task_type, "a domain expert")
    goal_text = goal.strip() if goal else "Deliver a high-quality answer that is accurate and practical."

    return (
        f"You are {role}.\n"
        f"Goal: {goal_text}\n"
        "Context: If assumptions are missing, list them first before solving.\n"
        f"Task: {prompt.strip()}\n"
        "Constraints:\n"
        "- Keep the answer focused and non-generic.\n"
        "- Use clear headings and bullets when useful.\n"
        "- If relevant, include one concrete example.\n"
        "Output format:\n"
        "1. Summary\n"
        "2. Detailed answer\n"
        "3. Quick self-check against the goal"
    )


def _instruction_adherence(prompt: str, response: str) -> float:
    score = 6.0
    lower_prompt = prompt.lower()
    lower_response = response.lower()

    wants_json = "json" in lower_prompt
    wants_bullets = any(w in lower_prompt for w in ["bullet", "list"])
    wants_steps = any(w in lower_prompt for w in ["step-by-step", "step by step", "steps"])
    wants_brief = any(w in lower_prompt for w in ["concise", "brief", "short"])

    if wants_json and ("{" in response and "}" in response):
        score += 1.5
    elif wants_json:
        score -= 1.5

    if wants_bullets and re.search(r"(^|\n)\s*[-*]\s+", response):
        score += 1.0
    elif wants_bullets:
        score -= 1.0

    if wants_steps and ("step 1" in lower_response or re.search(r"(^|\n)\s*\d+\.", response)):
        score += 1.0
    elif wants_steps:
        score -= 1.0

    word_count = len(response.split())
    if wants_brief and word_count <= 140:
        score += 1.0
    elif wants_brief and word_count > 180:
        score -= 1.0

    return round(_clip(score), 1)


def _relevance_score(scenario: str, response: str) -> float:
    scenario_tokens = _tokenize(scenario)
    if not scenario_tokens:
        return 5.0
    overlap = scenario_tokens & _tokenize(response)
    ratio = len(overlap) / max(1, len(scenario_tokens))
    return round(_clip(ratio * 10), 1)


def _structure_score(response: str) -> float:
    score = 4.0
    if re.search(r"(^|\n)\s*[-*]\s+", response):
        score += 2.0
    if re.search(r"(^|\n)\s*\d+\.", response):
        score += 2.0
    if len(response.splitlines()) > 2:
        score += 1.0
    return round(_clip(score), 1)


def _conciseness_score(response: str) -> float:
    words = len(response.split())
    if words < 40:
        return 5.0
    if words <= 220:
        return 8.5
    if words <= 320:
        return 7.0
    return 5.5


def run_prompt_test(scenario: str, prompts: list[str]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        combined_prompt = (
            f"{prompt.strip()}\n\n"
            "Use the following scenario for your answer.\n"
            f"Scenario: {scenario.strip()}"
        )
        routed = route_request(combined_prompt)
        response = routed["response"]

        relevance = _relevance_score(scenario, response)
        adherence = _instruction_adherence(prompt, response)
        structure = _structure_score(response)
        conciseness = _conciseness_score(response)

        overall = round(
            (0.4 * relevance) + (0.3 * adherence) + (0.2 * structure) + (0.1 * conciseness),
            1,
        )

        runs.append(
            {
                "index": idx,
                "prompt": prompt,
                "response": response,
                "score": overall,
                "metrics": {
                    "relevance": relevance,
                    "instruction_adherence": adherence,
                    "structure": structure,
                    "conciseness": conciseness,
                },
                "router_meta": {
                    "model": routed["model"],
                    "task_type": routed["task_type"],
                    "complexity_score": routed["complexity_score"],
                    "latency_sec": routed["latency_sec"],
                    "total_tokens": routed["total_tokens"],
                    "from_cache": routed["from_cache"],
                },
            }
        )

    ranked = sorted(runs, key=lambda row: row["score"], reverse=True)
    winner = ranked[0]["index"] if ranked else None

    return {
        "scenario": scenario,
        "winner_index": winner,
        "ranking": [row["index"] for row in ranked],
        "runs": runs,
    }
