import re
from typing import List, Set


NEGATIVE_PHRASE_PATTERN = re.compile(
    r"\b(?:not|except|excluding|other than|without|neither|never|outside of|avoid|avoids|avoiding|do not|don't|dislike|dislikes|hate|hates|allergic to|不要|避免|不吃|除了|不含)\s+([A-Za-z0-9_\u4e00-\u9fff\s]{2,30})",
    re.IGNORECASE
)

STOP_WORDS = {"the", "a", "an", "any", "that", "this", "which", "whose", "it", "they", "user", "person"}


def extract_negative_terms(query: str) -> List[str]:
    """Extract entities or attributes that the query explicitly forbids or excludes (EN & ZH)."""
    if not query:
        return []

    raw_matches = NEGATIVE_PHRASE_PATTERN.findall(query)
    clean_terms: List[str] = []

    for match in raw_matches:
        # Split by punctuation or conjunctions to avoid taking the rest of the sentence
        clause = re.split(r"[,;\.\?!，。；？！]|\b(?:and|or|but|where|when|who)\b", match)[0].strip().lower()
        words = [w for w in re.findall(r"[\w\u4e00-\u9fff]+", clause) if w not in STOP_WORDS]
        if words:
            clean_term = " ".join(words)
            if len(clean_term) >= 2:
                clean_terms.append(clean_term)

    return clean_terms


def compute_negative_penalty(content_lower: str, negative_terms: List[str]) -> float:
    """Compute penalty score for memories that match explicitly forbidden criteria."""
    if not negative_terms or not content_lower:
        return 0.0

    penalty = 0.0
    # If the memory explicitly represents a user constraint, allergy, or dietary restriction,
    # do not penalize it for mentioning the restricted item!
    is_constraint_doc = bool(re.search(r"\b(?:constraint|restriction|allergy|allergic|vegan|vegetarian|diet|kosher|halal)\b|\[strict constraint\]", content_lower))

    for term in negative_terms:
        if term in content_lower:
            if is_constraint_doc:
                continue

            # Check if the memory itself negates or records avoidance of this term
            # (e.g. 'not seafood', 'avoids dairy and seafood', 'never eats meat')
            neg_in_doc = bool(re.search(
                rf"\b(?:not|no|non|free from|without|avoid|avoids|avoiding|never|don't|dislike|dislikes|allergy|allergic|不吃|避免|不含)\s*(?:[A-Za-z0-9_\u4e00-\u9fff\s]{{0,25}}\s*)?{re.escape(term)}\b",
                content_lower
            ))
            if not neg_in_doc:
                # Violates negative constraint: apply substantial downweighting
                penalty += 0.40

    return min(0.80, penalty)
