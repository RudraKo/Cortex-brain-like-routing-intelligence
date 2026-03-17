import re
import hashlib
from functools import lru_cache
from typing import Optional
from .groq_client import generate_completion

# ── Model tiers ───────────────────────────────────────────────────────────────
FAST_MODEL   = "llama-3.1-8b-instant"
MEDIUM_MODEL = "llama-3.1-70b-versatile"
HEAVY_MODEL  = "llama-3.3-70b-versatile"

# ── Task types & their system prompts ─────────────────────────────────────────
TASK_SYSTEM_PROMPTS = {
    "coding": (
        "You are an expert software engineer. Provide clean, well-commented, "
        "production-quality code. Always include usage examples where relevant."
    ),
    "math": (
        "You are an expert mathematician. Solve problems step-by-step with "
        "clear reasoning. Show all intermediate steps and explain the logic."
    ),
    "creative": (
        "You are a creative writing expert. Produce vivid, original, and "
        "engaging content that shows imagination and strong narrative voice."
    ),
    "factual": (
        "You are a knowledgeable assistant. Answer concisely and accurately. "
        "Prefer short, factual responses. Cite key concepts when helpful."
    ),
}

# ── Simple prompt cache: { sha256_hash -> result_dict } ───────────────────────
_prompt_cache: dict = {}

def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()


# ── 1. Task Type Classifier ───────────────────────────────────────────────────
def classify_task(prompt: str) -> str:
    """Return one of: 'coding', 'math', 'creative', 'factual'."""
    lower = prompt.lower()

    coding_kw   = {"code","script","function","class","debug","refactor","implement",
                   "python","javascript","typescript","sql","api","algorithm","binary",
                   "array","tree","graph","react","django","fastapi","compile","runtime"}
    math_kw     = {"calculate","solve","prove","theorem","equation","integral","derivative",
                   "matrix","vector","probability","statistics","limit","sum","formula",
                   "trigonometry","algebra","geometry","calculus","optimize"}
    creative_kw = {"write","poem","story","essay","creative","fiction","narrative",
                   "metaphor","character","plot","scene","describe","imagine","draft"}

    words = set(re.findall(r'\b\w+\b', lower))

    coding_score   = len(words & coding_kw)
    math_score     = len(words & math_kw)
    creative_score = len(words & creative_kw)

    scores = {"coding": coding_score, "math": math_score, "creative": creative_score}
    best   = max(scores, key=scores.get)

    return best if scores[best] > 0 else "factual"


# ── 2. Complexity Analyzer (multi-factor, returns 0.0–1.0) ────────────────────
def analyze_complexity(prompt: str) -> float:
    """
    Multi-factor prompt complexity score between 0.0 and 1.0.
    0.0 = very simple / factual
    1.0 = highly complex / multi-step / long
    """
    score = 0.0
    lower = prompt.lower()

    # Factor 1 – Length
    words = len(prompt.split())
    if words > 150:
        score += 0.35
    elif words > 75:
        score += 0.20
    elif words > 30:
        score += 0.05

    # Factor 2 – Complex action keywords
    action_kw = {
        "analyze","compare","design","architect","implement","refactor","debug",
        "optimize","calculate","prove","derive","explain in detail","step by step"
    }
    if any(kw in lower for kw in action_kw):
        score += 0.20

    # Factor 3 – Technical domain density
    technical_kw = {
        "algorithm","complexity","recursion","dynamic programming","neural","transformer",
        "database","distributed","concurrency","async","multithread","microservice",
        "kubernetes","docker","machine learning","gradient","tensor","backpropagation"
    }
    tech_matches = sum(1 for kw in technical_kw if kw in lower)
    score += min(0.30, tech_matches * 0.10)

    # Factor 4 – Code blocks or mathematical symbols push complexity up
    if "```" in prompt:
        score += 0.20
    if re.search(r'[∑∫∏√∞≤≥∇∂]', prompt):
        score += 0.20
    elif re.search(r'[=\+\-\*\/\^%]', prompt) and score > 0.1:
        score += 0.05

    # Factor 5 – Multi-part questions (numbered lists, bullet points)
    if re.search(r'(\d+[\.\)] |\* |- )', prompt):
        score += 0.10

    return min(1.0, round(score, 3))


# ── 3. Three-Tier Router ──────────────────────────────────────────────────────
def select_model(score: float) -> str:
    """Map complexity score to model tier."""
    if score >= 0.55:
        return HEAVY_MODEL
    elif score >= 0.25:
        return MEDIUM_MODEL
    else:
        return FAST_MODEL


# ── 4. Quality Gatekeeper (auto-retry if fast model gives thin answer) ─────--
def _is_response_thin(text: str, min_words: int = 12) -> bool:
    return len(text.split()) < min_words


# ── 5. Main entry point ───────────────────────────────────────────────────────
def route_request(prompt: str) -> dict:
    """
    Full pipeline:
    1. Check cache.
    2. Classify task type → inject system prompt.
    3. Compute complexity score → select model tier.
    4. Generate response.
    5. Quality gate: retry with heavy model if response is too thin.
    6. Store result in cache and return.
    """
    key = _cache_key(prompt)
    if key in _prompt_cache:
        cached = dict(_prompt_cache[key])
        cached["from_cache"] = True
        return cached

    task_type   = classify_task(prompt)
    system_msg  = TASK_SYSTEM_PROMPTS[task_type]
    score       = analyze_complexity(prompt)
    model       = select_model(score)

    result = generate_completion(prompt, model, system_message=system_msg)
    retried = False

    # Quality gate: if fast model gives too-short answer, escalate to heavy
    if model == FAST_MODEL and _is_response_thin(result["response"]):
        result  = generate_completion(prompt, HEAVY_MODEL, system_message=system_msg)
        model   = HEAVY_MODEL
        retried = True

    result["complexity_score"] = score
    result["task_type"]        = task_type
    result["retried"]          = retried
    result["from_cache"]       = False

    # Cache the result
    _prompt_cache[key] = result
    return result
