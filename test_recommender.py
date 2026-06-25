"""
Test suite — Project 3 (improved)
Tests stop-word filtering, synonym mapping, fuzzy matching, and pipeline scores.
"""

from recommendation_engine import (
    RecommendationPipeline, sanitize_input, apply_synonyms,
    fuzzy_match_skill, display_recommendations, STOP_WORDS
)


def section(title):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def sub(title):
    print(f"\n  ── {title}")


# ─────────────────────────────────────────────────────────────────
def test_stop_word_filter():
    section("IMPROVEMENT 1 — Stop-Word Filtering")

    cases = [
        (["I know Python and basic Java"],          ["Python", "Java"]),
        (["I'm good at React and some Node.js"],    ["React", "Node.js"]),
        (["My skills are SQL, also MongoDB"],       ["SQL", "MongoDB"]),
        (["I have experience with AWS"],            ["AWS"]),
        (["Python"],                                ["Python"]),
        (["  Docker  ", "Kubernetes"],              ["Docker", "Kubernetes"]),
    ]

    all_pass = True
    for raw, expected_subset in cases:
        result = sanitize_input(raw)
        result_lower = [r.lower() for r in result]
        exp_lower    = [e.lower() for e in expected_subset]
        passed = all(e in result_lower for e in exp_lower)
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_pass = False
        print(f"\n  Input   : {raw}")
        print(f"  Output  : {result}")
        print(f"  Expected: {expected_subset}  →  {status}")

    print(f"\n  {'All stop-word tests passed ✓' if all_pass else 'Some tests FAILED ✗'}")


# ─────────────────────────────────────────────────────────────────
def test_synonym_mapping():
    section("IMPROVEMENT 2a — Synonym / Keyword Mapping")

    cases = [
        (["Frontend", "Developer"],             ["web design"]),
        (["K8s"],                               ["kubernetes"]),
        (["ML"],                                ["machine learning"]),
        (["DevOps"],                            ["ci/cd"]),
        (["REST", "API"],                       ["apis"]),
        (["Cloud", "Computing"],                ["aws"]),
        (["CI", "CD"],                          ["ci/cd"]),
        (["Data", "Science"],                   ["data analysis"]),
        (["Pytorch"],                           ["neural networks"]),
        (["Postgres"],                          ["postgresql"]),
    ]

    all_pass = True
    for tokens, expected_subset in cases:
        result = apply_synonyms(tokens)
        result_lower = [r.lower() for r in result]
        exp_lower    = [e.lower() for e in expected_subset]
        passed = all(e in result_lower for e in exp_lower)
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_pass = False
        print(f"\n  Input    : {tokens}")
        print(f"  Output   : {result}")
        print(f"  Expected : {expected_subset}  →  {status}")

    print(f"\n  {'All synonym tests passed ✓' if all_pass else 'Some tests FAILED ✗'}")


# ─────────────────────────────────────────────────────────────────
def test_fuzzy_matching():
    section("IMPROVEMENT 2b — Fuzzy / N-gram Matching")

    pipeline = RecommendationPipeline()
    pipeline.load_dataset("raw_skills.csv")
    vocab = pipeline.vectorizer.vocabulary

    cases = [
        ("Pythn",       "python"),
        ("javascrpit",  "javascript"),
        ("kubernets",   "kubernetes"),
        ("tensorfow",   "tensorflow"),
        ("databse",     "database design"),
        ("dockerr",     "docker"),
    ]

    all_pass = True
    for typo, expected in cases:
        matched = fuzzy_match_skill(typo, vocab, threshold=0.45)
        passed  = matched.lower() == expected.lower()
        status  = "✓ PASS" if passed else f"✗ FAIL (got '{matched}')"
        if not passed:
            all_pass = False
        print(f"\n  Typo      : {typo}")
        print(f"  Matched   : {matched}")
        print(f"  Expected  : {expected}  →  {status}")

    print(f"\n  {'All fuzzy tests passed ✓' if all_pass else 'Some tests FAILED ✗'}")


# ─────────────────────────────────────────────────────────────────
def test_pipeline_scores():
    section("PIPELINE — 5 Recommendation Scenarios")

    pipeline = RecommendationPipeline(top_n=3)
    pipeline.load_dataset("raw_skills.csv")

    scenarios = [
        ("Cloud & DevOps",        ["Docker", "Kubernetes", "AWS"]),
        ("Python / ML",           ["Python", "Machine Learning", "TensorFlow"]),
        ("Full Stack Web",        ["JavaScript", "React", "Node.js"]),
        ("Database Specialist",   ["SQL", "Database Design", "Performance Tuning"]),
        ("Natural Language Input",["I know Python and I have basic Cloud Computing and Kubernetes experience"]),
    ]

    for name, raw_skills in scenarios:
        sub(f"Scenario: {name}")
        try:
            recs = pipeline.recommend(raw_skills)
            display_recommendations(recs)
            assert all(score > 0.0 for _, score in recs), "All scores should be > 0"
            print("  ✓ All scores > 0")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")


# ─────────────────────────────────────────────────────────────────
def test_cosine_not_zero():
    section("BUG-FIX VERIFICATION — Cosine must NOT be 0.0")

    pipeline = RecommendationPipeline(top_n=3)
    pipeline.load_dataset("raw_skills.csv")

    inputs = [
        ["Python", "SQL", "Machine Learning"],
        ["Docker", "AWS", "Linux"],
        ["JavaScript", "React", "CSS"],
    ]
    passed = True
    for skills in inputs:
        recs = pipeline.recommend(skills)
        for role, score in recs:
            if score == 0.0:
                print(f"  ✗ FAIL — {role} got 0.0 for input {skills}")
                passed = False
            else:
                print(f"  ✓  {role}: {score:.4f}  (input: {skills})")
    print(f"\n  {'Bug-fix verified ✓' if passed else 'Bug still present ✗'}")


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  PROJECT 3 — IMPROVED TEST SUITE                            ║")
    print("║  DecodeLabs AI Engineering Internship (Batch 2026)          ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    test_stop_word_filter()
    test_synonym_mapping()
    test_fuzzy_matching()
    test_cosine_not_zero()
    test_pipeline_scores()

    print("\n" + "=" * 64)
    print("  ALL TESTS COMPLETE")
    print("=" * 64 + "\n")
