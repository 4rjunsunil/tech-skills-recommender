"""Job-title filtering and NLTK-based technical-skill matching."""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams

REQUIRED_COLUMNS = {"job_title", "job_description", "source_type"}
_TOKENIZER = RegexpTokenizer(r"[A-Za-z0-9]+(?:[+#.][A-Za-z0-9+#.]*)*")

TECHNICAL_SKILLS: dict[str, tuple[str, ...]] = {
    "Agile": ("agile", "agile methodology"),
    "Airflow": ("airflow", "apache airflow"),
    "Ansible": ("ansible",),
    "Angular": ("angular",),
    "Apache Spark": ("apache spark", "spark", "pyspark"),
    "ASP.NET": ("asp.net", "asp net"),
    "AWS": ("aws", "amazon web services"),
    "Azure": ("azure", "microsoft azure"),
    "Azure DevOps": ("azure devops",),
    "Bash": ("bash", "bash scripting"),
    "C#": ("c#", "c sharp"),
    "C++": ("c++", "cpp"),
    "CI/CD": ("ci cd", "cicd", "continuous integration", "continuous delivery"),
    "CSS": ("css",),
    "Django": ("django",),
    "Docker": ("docker",),
    "Elasticsearch": ("elasticsearch",),
    "ETL": ("etl", "extract transform load"),
    "Excel": ("excel", "microsoft excel"),
    "FastAPI": ("fastapi", "fast api"),
    "Flask": ("flask",),
    "GCP": ("gcp", "google cloud", "google cloud platform"),
    "Git": ("git",),
    "GitHub": ("github",),
    "GitLab": ("gitlab",),
    "Go": ("golang", "go programming", "go language"),
    "GraphQL": ("graphql",),
    "Hadoop": ("hadoop",),
    "HTML": ("html",),
    "Java": ("java",),
    "JavaScript": ("javascript", "js"),
    "Jenkins": ("jenkins",),
    "Kafka": ("kafka", "apache kafka"),
    "Kotlin": ("kotlin",),
    "Kubernetes": ("kubernetes", "k8s"),
    "Linux": ("linux",),
    "Machine Learning": ("machine learning", "ml models"),
    "Matplotlib": ("matplotlib",),
    "Microsoft SQL Server": ("microsoft sql server", "sql server", "mssql"),
    "MongoDB": ("mongodb", "mongo db"),
    "MySQL": ("mysql",),
    "Next.js": ("next.js", "nextjs"),
    "NLP": ("nlp", "natural language processing"),
    "Node.js": ("node.js", "nodejs"),
    "NumPy": ("numpy",),
    "Pandas": ("pandas",),
    "PHP": ("php",),
    "PostgreSQL": ("postgresql", "postgres"),
    "Power BI": ("power bi", "powerbi"),
    "PowerShell": ("powershell",),
    "PyTorch": ("pytorch",),
    "Python": ("python",),
    "R": ("r programming", "r language"),
    "React": ("react",),
    "React Native": ("react native",),
    "Redis": ("redis",),
    "REST APIs": ("rest api", "rest apis", "restful api", "restful apis"),
    "Ruby": ("ruby",),
    "Rust": ("rust",),
    "Scala": ("scala",),
    "scikit-learn": ("scikit learn", "sklearn"),
    "Snowflake": ("snowflake",),
    "Spring": ("spring", "spring framework"),
    "Spring Boot": ("spring boot",),
    "SQL": ("sql", "structured query language"),
    "Statistics": ("statistics", "statistical analysis"),
    "Tableau": ("tableau",),
    "TensorFlow": ("tensorflow",),
    "Terraform": ("terraform",),
    "TypeScript": ("typescript", "ts"),
    "Unit testing": ("unit testing", "unit tests"),
    "Vue.js": ("vue.js", "vuejs"),
    ".NET": ("dotnet", "dot net"),
    "Data visualization": ("data visualization", "data visualisation"),
}


def _normalize_phrase(phrase: str) -> str:
    return " ".join(token.casefold().rstrip(".") for token in _TOKENIZER.tokenize(phrase))


_SKILL_ALIASES = {
    _normalize_phrase(alias): skill
    for skill, aliases in TECHNICAL_SKILLS.items()
    for alias in aliases
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

    if re.fullmatch(r"(?:sde|swe)(?:[- ]?\d+)?", normalized_query):
        normalized_query = "software engineer"

    title_text = jobs["job_title"].fillna("").astype(str).str.casefold()
    contains_query = title_text.str.contains(re.escape(normalized_query), regex=True)
    return jobs[contains_query].copy()


def rank_skills(jobs: pd.DataFrame) -> pd.DataFrame:
    """Count recognized technical skills using NLTK-tokenized phrases."""
    if jobs.empty:
        return pd.DataFrame(columns=["skill", "job_count", "share_percent"])

    skill_counts: Counter[str] = Counter()
    description_count = 0
    max_ngram_size = max(len(alias.split()) for alias in _SKILL_ALIASES)

    for description in jobs["job_description"].fillna("").astype(str):
        description_count += 1
        phrases: set[str] = set()
        for segment in re.split(r"(?<=[,;.!?])\s+|\n+", description):
            tokens = [
                token.casefold().rstrip(".")
                for token in _TOKENIZER.tokenize(segment)
            ]
            phrases.update(
                " ".join(phrase)
                for size in range(1, min(max_ngram_size, len(tokens)) + 1)
                for phrase in ngrams(tokens, size)
            )

        skills_in_posting = {
            _SKILL_ALIASES[phrase]
            for phrase in phrases
            if phrase in _SKILL_ALIASES
        }
        skill_counts.update(skills_in_posting)

    results = pd.DataFrame(
        [
            {
                "skill": skill,
                "job_count": count,
                "share_percent": count / description_count * 100,
            }
            for skill, count in skill_counts.items()
        ]
    )
    if results.empty:
        return pd.DataFrame(columns=["skill", "job_count", "share_percent"])
    return results.sort_values(
        ["job_count", "skill"], ascending=[False, True], ignore_index=True
    )
