"""Job-title filtering and NLTK-based keyword extraction."""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams

REQUIRED_COLUMNS = {"job_title", "job_description", "source_type"}

_TOKENIZER = RegexpTokenizer(r"[A-Za-z0-9]+(?:[+#.][A-Za-z0-9+#.]*)*")
_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "business",
    "create",
    "customer",
    "develop",
    "findings",
    "for",
    "from",
    "insights",
    "in",
    "into",
    "is",
    "it",
    "job",
    "large",
    "maintain",
    "operational",
    "of",
    "on",
    "or",
    "our",
    "prepare",
    "project",
    "projects",
    "provide",
    "reliable",
    "role",
    "share",
    "stakeholder",
    "stakeholders",
    "support",
    "team",
    "teams",
    "the",
    "their",
    "then",
    "this",
    "to",
    "through",
    "use",
    "using",
    "with",
    "write",
    "query",
    "queries",
}
_STOP_WORDS.update(
    {
        "all",
        "also",
        "any",
        "applicant",
        "applicants",
        "application",
        "applications",
        "because",
        "been",
        "before",
        "being",
        "both",
        "but",
        "can",
        "candidate",
        "candidates",
        "could",
        "degree",
        "degrees",
        "did",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "education",
        "employer",
        "employers",
        "excellent",
        "few",
        "further",
        "had",
        "has",
        "having",
        "he",
        "her",
        "here",
        "him",
        "his",
        "how",
        "if",
        "its",
        "may",
        "me",
        "more",
        "most",
        "must",
        "need",
        "needs",
        "nor",
        "not",
        "now",
        "off",
        "once",
        "only",
        "other",
        "ourselves",
        "out",
        "over",
        "own",
        "please",
        "position",
        "positions",
        "preferred",
        "provides",
        "provided",
        "qualifications",
        "qualified",
        "requirements",
        "responsibilities",
        "same",
        "seeking",
        "she",
        "should",
        "some",
        "such",
        "than",
        "that",
        "their",
        "theirs",
        "them",
        "themselves",
        "there",
        "these",
        "they",
        "those",
        "too",
        "under",
        "until",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "analyst",
        "analysts",
        "data",
        "developer",
        "developers",
        "engineer",
        "engineers",
        "scientist",
        "scientists",
        "software",
        "skill",
        "skills",
        "year",
        "years",
        "working",
        "ability",
        "able",
        "have",
        "including",
        "management",
        "opportunity",
        "related",
        "time",
        "company",
        "companies",
        "tools",
        "benefit",
        "benefits",
        "development",
        "technical",
        "bachelor",
        "information",
        "identify",
        "professional",
        "environment",
        "analytical",
        "within",
        "equal",
        "employment",
        "disability",
        "status",
        "protected",
        "veteran",
        "color",
        "race",
        "sex",
        "sexual",
        "orientation",
        "gender",
        "identity",
        "national",
        "origin",
        "religion",
        "regard",
        "considered",
        "accommodation",
        "applicable",
        "without",
        "based",
        "location",
        "range",
        "required",
        "requirements",
        "strong",
        "knowledge",
        "process",
        "processes",
        "problem",
        "problems",
        "solution",
        "solutions",
        "include",
        "includes",
        "understanding",
        "us",
        "perform",
        "performance",
        "science",
        "ensure",
        "help",
        "people",
        "quality",
        "technology",
        "best",
        "computer",
        "opportunities",
        "review",
        "well",
        "communication",
        "communications",
        "interpersonal",
        "reporting",
        "system",
        "systems",
        "functional",
        "key",
        "looking",
        "organization",
        "relevant",
        "across",
        "day",
        "description",
        "internal",
        "do",
        "e.g",
        "level",
        "multiple",
        "office",
        "one",
        "services",
        "building",
        "life",
        "sources",
        "specific",
        "up",
        "apply",
        "complex",
        "critical",
        "developing",
        "engineering",
        "growth",
        "industry",
        "making",
        "medical",
        "salary",
        "strategic",
        "understand",
        "vision",
        "client",
        "closely",
        "compensation",
        "drive",
        "driven",
        "effectively",
        "employee",
        "employees",
        "join",
        "written",
        "age",
        "responsible",
        "access",
        "customers",
        "dental",
        "improve",
        "solving",
        "various",
        "advanced",
        "improvement",
        "initiatives",
        "issues",
        "plan",
        "plans",
        "providing",
        "re",
        "etc",
        "non",
        "full",
        "high",
        "world",
        "technologies",
        "product",
        "part",
        "partners",
        "goals",
        "lead",
        "leadership",
        "solve",
        "stories",
    }
)
_SINGLE_WORD_STOP_WORDS = _STOP_WORDS | {
    "analyse",
    "analysis",
    "analyze",
    "bi",
    "build",
    "clean",
    "cleaning",
    "dataset",
    "explain",
    "data",
    "experience",
    "new",
    "power",
    "present",
    "report",
    "reports",
    "senior",
    "work",
    "dataset",
    "datasets",
    "insight",
    "microsoft",
    "machine",
    "learning",
}
_DISPLAY_NAMES = {
    "api": "API",
    "apis": "APIs",
    "aws": "AWS",
    "c#": "C#",
    "c++": "C++",
    "etl": "ETL",
    "excel": "Excel",
    "github": "GitHub",
    "javascript": "JavaScript",
    "linux": "Linux",
    "mongodb": "MongoDB",
    "nlp": "NLP",
    "nosql": "NoSQL",
    "pandas": "Pandas",
    "postgresql": "PostgreSQL",
    "power bi": "Power BI",
    "pytorch": "PyTorch",
    "python": "Python",
    "rest api": "REST API",
    "rest apis": "REST APIs",
    "r": "R",
    "sql": "SQL",
    "tableau": "Tableau",
    "tensorflow": "TensorFlow",
}


def clean_job_ads(jobs: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the CSV columns used by the app."""
    missing_columns = REQUIRED_COLUMNS.difference(jobs.columns)
    if missing_columns:
        raise ValueError(
            "The job ads CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    cleaned = jobs.copy()
    cleaned["job_title"] = cleaned["job_title"].fillna("").astype(str).str.strip()
    cleaned["job_description"] = (
        cleaned["job_description"].fillna("").astype(str).str.strip()
    )
    cleaned["source_type"] = cleaned["source_type"].fillna("").astype(str).str.strip()
    return cleaned[
        cleaned["job_title"].ne("") & cleaned["job_description"].ne("")
    ].reset_index(drop=True)


def filter_jobs_by_title(jobs: pd.DataFrame, query: str) -> pd.DataFrame:
    """Return postings whose title contains the user's title query."""
    normalized_query = " ".join(query.casefold().split())
    if not normalized_query:
        return jobs.iloc[0:0].copy()

    title_text = jobs["job_title"].fillna("").astype(str).str.casefold()
    contains_query = title_text.str.contains(re.escape(normalized_query), regex=True)
    return jobs[contains_query].copy()


def _display_keyword(keyword: str) -> str:
    display_name = _DISPLAY_NAMES.get(keyword)
    if display_name:
        return display_name
    return " ".join(token.capitalize() for token in keyword.split())


def rank_skills(jobs: pd.DataFrame) -> pd.DataFrame:
    """Extract NLTK unigram, bigram, and trigram keywords by posting frequency."""
    if jobs.empty:
        return pd.DataFrame(columns=["skill", "job_count", "share_percent"])

    keyword_document_counts: Counter[str] = Counter()
    document_keywords: list[set[str]] = []

    for description in jobs["job_description"].fillna("").astype(str):
        keywords: set[str] = set()
        for segment in re.split(r"(?<=[,;.!?])\s+|\n+", description):
            tokens = [
                token.casefold().rstrip(".")
                for token in _TOKENIZER.tokenize(segment)
                if token.casefold().rstrip(".")
            ]
            keywords.update(
                " ".join(term)
                for size in range(1, 4)
                for term in ngrams(tokens, size)
                if (
                    (
                        term[0] not in _SINGLE_WORD_STOP_WORDS
                        if size == 1
                        else not any(token in _STOP_WORDS for token in term)
                    )
                    and not (size == 1 and term[0].isdigit())
                    and not (
                        size == 1
                        and len(term[0]) == 1
                        and term[0] not in {"r"}
                    )
                )
            )
        document_keywords.append(keywords)
        keyword_document_counts.update(keywords)

    filtered_counts: FreqDist[str] = FreqDist()
    for keyword, count in keyword_document_counts.items():
        if count >= 2:
            filtered_counts[keyword] = count

    results = pd.DataFrame(
        [
            {
                "skill": _display_keyword(keyword),
                "job_count": count,
                "share_percent": count / len(document_keywords) * 100,
            }
            for keyword, count in filtered_counts.items()
        ]
    )
    if results.empty:
        return pd.DataFrame(columns=["skill", "job_count", "share_percent"])
    return results.sort_values(
        ["job_count", "skill"], ascending=[False, True], ignore_index=True
    )
