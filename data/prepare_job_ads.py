"""Prepare an attributed, role-focused subset from the public source CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

SOURCE_TYPE = "xanderios/linkedin-job-postings on Hugging Face (MIT)"
MAX_POSTINGS_PER_ROLE = 80
RANDOM_SEED = 42

ROLE_PATTERNS = {
    "Data Scientist": re.compile(r"\bdata scientist\b", re.IGNORECASE),
    "Data Analyst": re.compile(r"\bdata analyst\b", re.IGNORECASE),
    "Data Engineer": re.compile(r"\bdata engineer\b", re.IGNORECASE),
    "Software Engineer": re.compile(r"\bsoftware engineer\b", re.IGNORECASE),
    "Software Developer": re.compile(r"\bsoftware developer\b", re.IGNORECASE),
    "Product Analyst": re.compile(r"\bproduct analyst\b", re.IGNORECASE),
}
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\w)"
)


def roles_for_title(title: object) -> list[str]:
    title_text = str(title or "")
    return [
        role for role, pattern in ROLE_PATTERNS.items() if pattern.search(title_text)
    ]


def clean_description(description: object) -> str:
    text = str(description or "")
    text = EMAIL_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = PHONE_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_dataset(source_path: Path, output_path: Path) -> pd.DataFrame:
    """Select a reproducible sample of real technology job postings."""
    jobs = pd.read_csv(
        source_path,
        usecols=["job_id", "title", "description"],
        on_bad_lines="skip",
    )
    jobs["job_title"] = jobs["title"].fillna("").astype(str).str.strip()
    jobs["job_description"] = jobs["description"].fillna("").map(clean_description)
    jobs = jobs[
        jobs["job_title"].ne("")
        & jobs["job_description"].ne("")
        & jobs["job_description"].str.len().ge(80)
    ].copy()
    jobs["dataset_roles"] = jobs["job_title"].map(roles_for_title)
    jobs = jobs[jobs["dataset_roles"].map(bool)].drop_duplicates(subset=["job_id"])

    samples = []
    for role in ROLE_PATTERNS:
        group = jobs[jobs["dataset_roles"].map(lambda roles: role in roles)]
        samples.append(
            group.sample(
                n=min(len(group), MAX_POSTINGS_PER_ROLE),
                random_state=RANDOM_SEED,
            )
        )
    subset = pd.concat(samples, ignore_index=True)
    subset = subset.drop_duplicates(subset=["job_id"]).sort_values(
        "job_title", ignore_index=True
    )
    result = subset.assign(source_type=SOURCE_TYPE)[
        ["job_title", "job_description", "source_type"]
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", type=Path, help="Path to downloaded source CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("job_ads.csv"),
        help="Output CSV path (default: data/job_ads.csv)",
    )
    arguments = parser.parse_args()

    result = prepare_dataset(arguments.source_csv, arguments.output)
    print(f"Wrote {len(result)} postings to {arguments.output}")
    print("Postings per role:")
    for role in ROLE_PATTERNS:
        count = result["job_title"].map(
            lambda title: role in roles_for_title(title)
        ).sum()
        print(f"  {role}: {count}")


if __name__ == "__main__":
    main()
