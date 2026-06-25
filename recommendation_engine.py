"""
Project 3: AI Recommendation Logic — Tech Stack Recommender
DecodeLabs AI Engineering Internship (Batch 2026)

Improvements shipped in this version:
  1. Stop-word filtering      — strips "I", "know", "and", "basic" etc.
  2. Synonym / N-gram mapping — "K8s"→kubernetes, "Frontend Developer"→web design
  3. Fuzzy bigram matching    — catches near-typos like "kubernets"
  4. FastAPI backend          — HTML reads live CSV via HTTP
  5. Case-normalisation fix   — eliminates the cosine-always-0 bug
"""

import math
import re
from collections import defaultdict
from typing import List, Dict, Tuple
import csv

# ─────────────────────────────────────────────────────────────────
# KNOWN MULTI-WORD SKILLS (preserved during sentence-mode parsing)
# ─────────────────────────────────────────────────────────────────
KNOWN_BIGRAMS = {
    "machine learning", "database design", "cloud computing",
    "deep learning", "neural networks", "spring boot",
    "performance tuning", "web design", "computer vision",
    "network security", "system design", "data analysis",
    "linear algebra", "cost optimization", "query optimization",
    "data processing", "model optimization", "backup recovery",
    "data security", "vulnerability assessment", "penetration testing",
    "risk management", "architecture patterns", "message queues",
    "responsive design", "version control", "ci/cd",
    "infrastructure as code", "incident response",
    "performance optimization", "database administration",
    "algorithm design", "node.js", "vue.js",
}

# ─────────────────────────────────────────────────────────────────
# IMPROVEMENT 1 — STOP-WORD FILTER
# ─────────────────────────────────────────────────────────────────
STOP_WORDS = {
    "i", "i'm", "im", "know", "knowing", "knows",
    "and", "or", "the", "a", "an", "of", "in", "to", "for",
    "with", "have", "has", "am", "is", "are", "was", "were",
    "be", "been", "being", "do", "does", "did",
    "can", "could", "would", "should", "will", "may", "might",
    "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those",
    "some", "any", "few", "more", "most", "other", "such",
    "no", "not", "only", "same", "so", "than", "too", "very",
    "basic", "advanced", "expert", "beginner", "intermediate",
    "good", "great", "strong", "little", "bit",
    "also", "well", "like", "about", "around", "experience",
    "knowledge", "understanding", "skills", "skill",
    "technology", "technologies", "tool", "tools",
    "work", "working", "use", "using", "used",
    "at", "on", "by", "from", "up", "out", "as", "if",
}


def _reconstruct_bigrams(tokens: List[str]) -> List[str]:
    """
    After sentence-mode splitting, merge adjacent tokens that form a known
    multi-word skill (e.g. ["Machine","Learning"] → ["Machine Learning"]).
    """
    result, i = [], 0
    lc = [t.lower() for t in tokens]
    while i < len(lc):
        if i + 1 < len(lc) and (lc[i] + " " + lc[i + 1]) in KNOWN_BIGRAMS:
            result.append(tokens[i] + " " + tokens[i + 1])
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


def sanitize_input(raw_skills: List[str]) -> List[str]:
    """
    Improvement 1: Stop-word filter + text sanitisation.

    Two modes are detected automatically per list element:
      • Skill-list mode  — "Machine Learning", "Node.js"  → kept as-is
      • Sentence mode    — "I know Python and basic Java" → tokenised & filtered

    A sentence is detected when ANY word in the element is a stop word.
    """
    out = []
    for raw in raw_skills:
        # Split on commas first (handles "Python, Cloud" in a single string)
        parts = re.split(r"[,]+", raw)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            words = part.split()
            has_stop = any(w.lower() in STOP_WORDS for w in words)

            if has_stop:
                # ── Sentence mode: tokenise, drop stop words ──
                cleaned_tokens = []
                for tok in words:
                    tok = tok.strip("\"'()[]{}!?;:")
                    if tok and tok.lower() not in STOP_WORDS and len(tok) >= 2:
                        cleaned_tokens.append(tok)
                # Reconstruct any broken known bigrams
                out.extend(_reconstruct_bigrams(cleaned_tokens))
            else:
                # ── Skill-list mode: keep multi-word skill intact ──
                tok_clean = part.strip("\"'()[]{}!?;:")
                if tok_clean and len(tok_clean) >= 2:
                    out.append(tok_clean)
    return out


# ─────────────────────────────────────────────────────────────────
# IMPROVEMENT 2a — SYNONYM / KEYWORD MAPPING
# ─────────────────────────────────────────────────────────────────
SYNONYM_MAP: Dict[str, str] = {
    # Frontend
    "frontend developer": "web design", "front-end": "web design",
    "front end": "web design", "ui developer": "web design",
    "ui/ux": "web design",
    "reactjs": "react", "react.js": "react", "react js": "react",
    "vuejs": "vue.js", "vue js": "vue.js",
    "angularjs": "angular",
    "nextjs": "react", "next.js": "react",
    # Backend
    "backend developer": "apis", "back-end": "apis",
    "back end": "apis",
    "nodejs": "node.js", "node js": "node.js",
    "expressjs": "node.js",
    "springboot": "spring boot", "spring-boot": "spring boot",
    "fastapi": "python", "flask": "python", "django": "python",
    # Data / ML
    "ml": "machine learning", "dl": "neural networks",
    "ai": "machine learning",
    "artificial intelligence": "machine learning",
    "nlp": "machine learning",
    "data science": "data analysis", "data engineering": "data analysis",
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn",
    "tf": "tensorflow",
    "pytorch": "neural networks", "keras": "tensorflow",
    "cv": "computer vision", "deep learning": "neural networks",
    # DevOps / Cloud
    "devops": "ci/cd", "k8s": "kubernetes", "kube": "kubernetes",
    "aws cloud": "aws", "amazon web services": "aws",
    "gcp": "google cloud", "google cloud platform": "google cloud",
    "azure cloud": "azure", "microsoft azure": "azure",
    "iac": "terraform", "infrastructure as code": "terraform",
    "github actions": "ci/cd", "gitlab ci": "ci/cd",
    "cicd": "ci/cd", "ci cd": "ci/cd",
    "shell scripting": "bash", "shell": "bash",
    "cloud computing": "aws", "cloud": "aws",
    # Database
    "database": "database design", "databases": "database design",
    "rdbms": "sql", "postgres": "postgresql", "mongo": "mongodb",
    "nosql": "mongodb", "mysql database": "mysql",
    # Security
    "cybersec": "cybersecurity", "infosec": "cybersecurity",
    "pentesting": "penetration testing", "pen testing": "penetration testing",
    "appsec": "cybersecurity",
    # General
    "oop": "java", "object oriented": "java",
    "git version control": "git", "version control": "git",
    "rest api": "apis", "restful": "apis", "rest": "apis",
    "graphql api": "graphql", "microservice": "microservices",
    "automation testing": "automation", "test automation": "automation",
}


def apply_synonyms(skills: List[str]) -> List[str]:
    """
    Improvement 2a: Map user phrases to canonical dataset tags.
    Checks bigrams first (two adjacent tokens), then single tokens.
    """
    result, i = [], 0
    lc = [s.lower() for s in skills]
    while i < len(lc):
        if i + 1 < len(lc):
            bigram = lc[i] + " " + lc[i + 1]
            if bigram in SYNONYM_MAP:
                result.append(SYNONYM_MAP[bigram])
                i += 2
                continue
        result.append(SYNONYM_MAP.get(lc[i], lc[i]))
        i += 1
    return result


# ─────────────────────────────────────────────────────────────────
# IMPROVEMENT 2b — FUZZY / N-GRAM MATCHING
# ─────────────────────────────────────────────────────────────────
def _bigrams(s: str) -> set:
    s = s.lower().replace(" ", "_")
    if len(s) < 2:
        return {s}
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _bigram_jaccard(a: str, b: str) -> float:
    sa, sb = _bigrams(a), _bigrams(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def fuzzy_match_skill(skill: str, vocabulary: set,
                      threshold: float = 0.50) -> str:
    """
    Improvement 2b: Find the closest vocabulary term by character-bigram
    Jaccard similarity. Handles typos like "kubernets" → "kubernetes".
    """
    if skill in vocabulary:
        return skill
    best, best_score = skill, 0.0
    for term in vocabulary:
        score = _bigram_jaccard(skill, term)
        if score > best_score:
            best_score, best = score, term
    return best if best_score >= threshold else skill


# ─────────────────────────────────────────────────────────────────
# TF-IDF VECTORIZER  (all terms stored/matched in lowercase)
# ─────────────────────────────────────────────────────────────────
class TFIDFVectorizer:
    def __init__(self):
        self.vocabulary: set = set()
        self.idf_scores: Dict[str, float] = {}

    def _norm(self, terms: List[str]) -> List[str]:
        return [t.lower().strip() for t in terms]

    def fit(self, documents: List[List[str]]) -> None:
        normed = [self._norm(d) for d in documents]
        for doc in normed:
            self.vocabulary.update(doc)
        total = len(normed)
        tdc: Dict[str, int] = defaultdict(int)
        for doc in normed:
            for t in set(doc):
                tdc[t] += 1
        for t in self.vocabulary:
            self.idf_scores[t] = math.log(total / tdc[t]) if tdc[t] else 0.0

    def transform(self, document: List[str]) -> Dict[str, float]:
        normed = self._norm(document)
        total = len(normed)
        tc: Dict[str, int] = defaultdict(int)
        for t in normed:
            tc[t] += 1
        vec: Dict[str, float] = {}
        for t, c in tc.items():
            idf = self.idf_scores.get(t, 0.0)
            if idf > 0:
                vec[t] = (c / total) * idf
        return vec


# ─────────────────────────────────────────────────────────────────
# COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────────
class CosineSimilarityMatcher:
    @staticmethod
    def _mag(v: Dict[str, float]) -> float:
        return math.sqrt(sum(x * x for x in v.values()))

    @classmethod
    def cosine_similarity(cls, a: Dict[str, float],
                          b: Dict[str, float]) -> float:
        dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in set(a) | set(b))
        ma, mb = cls._mag(a), cls._mag(b)
        if ma == 0.0 or mb == 0.0:
            return 0.0
        return max(-1.0, min(1.0, dot / (ma * mb)))


# ─────────────────────────────────────────────────────────────────
# 4-STEP RECOMMENDATION PIPELINE
# ─────────────────────────────────────────────────────────────────
class RecommendationPipeline:
    def __init__(self, top_n: int = 3):
        self.top_n = top_n
        self.vectorizer = TFIDFVectorizer()
        self.matcher = CosineSimilarityMatcher()
        self.job_roles: Dict[str, List[str]] = {}
        self.job_vectors: Dict[str, Dict[str, float]] = {}

    def load_dataset(self, csv_path: str) -> None:
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                role = row["role"].strip()
                skills = [s.strip() for s in row["skills"].split(",")]
                self.job_roles[role] = skills
        print(f"✓ Loaded {len(self.job_roles)} job roles")
        self.vectorizer.fit(list(self.job_roles.values()))
        for role, skills in self.job_roles.items():
            self.job_vectors[role] = self.vectorizer.transform(skills)

    # ── Step 1: Ingestion ────────────────────────────────────────
    def step_1_ingest(self, raw_skills: List[str]) -> Dict[str, float]:
        cleaned   = sanitize_input(raw_skills)
        synonymed = apply_synonyms(cleaned)
        final     = [fuzzy_match_skill(s, self.vectorizer.vocabulary)
                     for s in synonymed]

        print(f"\n[STEP 1: INGESTION]")
        print(f"  Raw    : {raw_skills}")
        print(f"  Cleaned: {cleaned}")
        print(f"  Syns   : {synonymed}")
        print(f"  Fuzzy  : {final}")

        if len(final) < 3:
            raise ValueError(
                f"Only {len(final)} valid skill(s) detected after filtering. "
                "Please add more specific technical skills."
            )
        return self.vectorizer.transform(final)

    # ── Step 2: Scoring ──────────────────────────────────────────
    def step_2_score(self, user_vec: Dict[str, float]) -> Dict[str, float]:
        scores = {role: self.matcher.cosine_similarity(user_vec, jv)
                  for role, jv in self.job_vectors.items()}
        print(f"[STEP 2: SCORING]  Scored {len(scores)} roles")
        return scores

    # ── Step 3: Sorting ──────────────────────────────────────────
    def step_3_sort(self, scores: Dict[str, float]) -> List[Tuple[str, float]]:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"[STEP 3: SORTING]  Best → {ranked[0][0]} ({ranked[0][1]:.4f})")
        return ranked

    # ── Step 4: Filtering ────────────────────────────────────────
    def step_4_filter(self, ranked: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        top = ranked[: self.top_n]
        print(f"[STEP 4: FILTERING] Returning Top-{self.top_n}")
        return top

    def recommend(self, raw_skills: List[str]) -> List[Tuple[str, float]]:
        print("\n" + "=" * 60)
        print("RECOMMENDATION PIPELINE")
        print("=" * 60)
        uv   = self.step_1_ingest(raw_skills)
        sc   = self.step_2_score(uv)
        rk   = self.step_3_sort(sc)
        top  = self.step_4_filter(rk)
        print("=" * 60)
        return top


def display_recommendations(recommendations: List[Tuple[str, float]]) -> None:
    print("\n📊 TOP RECOMMENDED CAREER PATHS")
    print("-" * 60)
    for rank, (role, score) in enumerate(recommendations, 1):
        pct = score * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\n{rank}. {role.upper()}")
        print(f"   Match  : {pct:.1f}%  [{bar}]")
        print(f"   Cosine : {score:.4f}")


# ─────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Tech Stack Recommender — DecodeLabs Batch 2026")
    print("=" * 60)
    pipeline = RecommendationPipeline(top_n=3)
    pipeline.load_dataset("raw_skills.csv")
    print("\n💡 Enter skills (comma-separated, natural language OK):")
    print("   e.g.  Python, I know basic Cloud Computing and K8s")
    raw = input("Your skills: ").strip()
    skills = [s.strip() for s in raw.split(",") if s.strip()]
    try:
        recs = pipeline.recommend(skills)
        display_recommendations(recs)
    except ValueError as e:
        print(f"\n⚠️  {e}")
