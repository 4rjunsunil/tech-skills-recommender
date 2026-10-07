"""Job-title filtering and dictionary-based skill frequency analysis."""

from __future__ import annotations

import re

import pandas as pd

SKILL_PATTERNS: dict[str, tuple[str, ...]] = {
    "AWS": ("AWS", "Amazon Web Services"),
    "Azure": ("Azure", "Microsoft Azure"),
    "C++": ("C++",),
    "C#": ("C#", "C sharp"),
    "Communication": ("communication", "communicate"),
    "Data analysis": ("data analysis", "analyze data", "analyse data"),
    "Data cleaning": ("data cleaning", "cleaning data", "data cleansing"),
    "Data visualization": ("data visualization", "data visualisation", "data viz"),
    "Excel": ("Excel", "Microsoft Excel"),
    "Git": ("Git", "GitHub", "GitLab"),
    "Java": ("Java",),
    "JavaScript": ("JavaScript",),
    "Machine learning": ("machine learning", "ML models"),
    "Matplotlib": ("Matplotlib",),
    "Power BI": ("Power BI", "PowerBI"),
    "NLP": ("natural language processing", "NLP"),
    "NoSQL": ("NoSQL", "MongoDB", "Cassandra"),
    "Pandas": ("Pandas",),
    "PostgreSQL": ("PostgreSQL", "Postgres"),
    "Python": ("Python",),
    "PyTorch": ("PyTorch",),
    "R": ("R programming", "R language"),
    "REST APIs": ("REST API", "RESTful API", "REST APIs"),
    "scikit-learn": ("scikit-learn", "sklearn"),
    "SQL": ("SQL",),
    "Statistics": ("statistics", "statistical analysis"),
    "Tableau": ("Tableau",),
    "TensorFlow": ("TensorFlow",),
    "Testing": ("unit testing", "automated testing", "test automation"),
    "Docker": ("Docker",),
    "Kubernetes": ("Kubernetes",),
    "Linux": ("Linux",),
    "Spark": ("Apache Spark", "PySpark", "Spark"),
}


def _compile_skill_patterns() -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    for skill, aliases in SKILL_PATTERNS.items():
        alternatives = "|".join(
            sorted((re.escape(alias) for alias in aliases), key=len, reverse=True)
        )
        patterns[skill] = re.compile(
            rf"(?<![A-Za-z0-9])(?:{alternatives})(?![A-Za-z0-9])",
            flags=re.IGNORECASE,
        )
    return patterns


_COMPILED_SKILLS = _compile_skill_patterns()


def filter_jobs_by_title(jobs: pd.DataFrame, query: str) -> pd.DataFrame:
    """Return postings whose title contains the user's title query."""
    normalized_query = " ".join(query.casefold().split())
    if not normalized_query:
        return jobs.iloc[0:0].copy()

    title_text = jobs["job_title"].fillna("").astype(str).str.casefold()
    contains_query = title_text.str.contains(re.escape(normalized_query), regex=True)
    query_terms = set(normalized_query.split())
    contains_all_terms = title_text.map(
        lambda title: query_terms.issubset(set(title.split()))
    )
    return jobs[contains_query | contains_all_terms].copy()


def rank_skills(jobs: pd.DataFrame) -> pd.DataFrame:
    """Rank skills by the number of matching job postings, highest first."""
    if jobs.empty:
        return pd.DataFrame(columns=["skill", "job_count", "share_percent"])

    descriptions = jobs["job_description"].fillna("").astype(str)
    counts = {
        skill: int(descriptions.map(lambda text: bool(pattern.search(text))).sum())
        for skill, pattern in _COMPILED_SKILLS.items()
    }
    results = pd.DataFrame(
        [
            {"skill": skill, "job_count": count, "share_percent": count / len(jobs) * 100}
            for skill, count in counts.items()
            if count
        ]
    )
    if results.empty:
        return pd.DataFrame(columns=["skill", "job_count", "share_percent"])
    return results.sort_values(
        ["job_count", "skill"], ascending=[False, True], ignore_index=True
    )
