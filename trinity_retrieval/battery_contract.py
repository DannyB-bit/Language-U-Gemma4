"""Shared battery accounting: only explicit expected topics are successes."""


def assess_case(expected, kind, hit, weak_hit):
    """Return (bucket, diagnostic); admitted known false positives always fail."""
    got = hit or weak_hit
    if kind in ("miss", "kinship-fp", "weak-admit-fp"):
        if got is None:
            return "miss", None
        return "fail", f"noise/false positive admitted: {got['slug']}"
    accepted = set(expected.split("|")) if expected else set()
    if hit is not None and hit.get("roadmap") != "source" and hit["slug"] in accepted:
        return "strong", None
    if weak_hit is not None and weak_hit.get("roadmap") != "source" and weak_hit["slug"] in accepted:
        return "weak", None
    if kind == "miss-by-design" and got is None:
        return "miss", None
    return "fail", f"got {got['slug'] if got else 'MISS'} want {expected}"
