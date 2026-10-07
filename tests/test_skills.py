import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from data.prepare_job_ads import prepare_dataset
from src.exports import build_pdf_report
from src.skills import clean_job_ads, filter_jobs_by_title, rank_skills


class SkillAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.jobs = pd.DataFrame(
            [
                {
                    "job_title": "Data Analyst",
                    "job_description": "Use Python, SQL, Python, and Power BI.",
                },
                {
                    "job_title": "Senior Data Analyst",
                    "job_description": "Use Python and Excel.",
                },
                {
                    "job_title": "Software Engineer",
                    "job_description": "Use JavaScript and REST APIs.",
                },
                {
                    "job_title": "Analyst - Data",
                    "job_description": "Analyze data using SQL.",
                },
            ]
        )

    def test_title_search_matches_role_variants_case_insensitively(self):
        matches = filter_jobs_by_title(self.jobs, "data analyst")
        self.assertEqual(len(matches), 2)
        self.assertNotIn("Analyst - Data", matches["job_title"].tolist())

    def test_keyword_counts_are_per_job_posting_not_raw_mentions(self):
        results = rank_skills(self.jobs.iloc[:2])
        counts = results.set_index("skill")["job_count"]
        self.assertEqual(counts["Python"], 2)
        self.assertNotIn("SQL", counts.index)
        self.assertNotIn("Power BI", counts.index)

    def test_empty_search_and_empty_results_are_safe(self):
        self.assertTrue(filter_jobs_by_title(self.jobs, " ").empty)
        self.assertTrue(rank_skills(self.jobs.iloc[0:0]).empty)

    def test_nltk_extraction_can_surface_terms_not_in_a_skill_dictionary(self):
        jobs = pd.DataFrame(
            [
                {"job_description": "Build Snowflake data pipelines."},
                {"job_description": "Maintain Snowflake pipelines with SQL."},
            ]
        )
        results = rank_skills(jobs)
        counts = results.set_index("skill")["job_count"]
        self.assertEqual(counts["Snowflake"], 2)
        self.assertEqual(counts["Pipelines"], 2)
        self.assertNotIn("Will", counts.index)

    def test_multiword_technical_keyword_is_kept_as_a_phrase(self):
        jobs = pd.DataFrame(
            [
                {"job_description": "Use machine learning with Python."},
                {"job_description": "Build machine learning models."},
            ]
        )
        results = rank_skills(jobs)
        keywords = set(results["skill"])
        self.assertIn("Machine Learning", keywords)
        self.assertNotIn("Machine", keywords)
        self.assertNotIn("Learning", keywords)

    def test_cleaning_trims_text_and_drops_empty_required_values(self):
        raw = pd.DataFrame(
            [
                {
                    "job_title": " Data Analyst ",
                    "job_description": " Use SQL. ",
                    "source_type": " demo ",
                },
                {"job_title": "", "job_description": "Use Python.", "source_type": "demo"},
                {"job_title": "Developer", "job_description": " ", "source_type": "demo"},
            ]
        )
        cleaned = clean_job_ads(raw)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["job_title"], "Data Analyst")
        self.assertEqual(cleaned.iloc[0]["job_description"], "Use SQL.")

    def test_missing_dataset_columns_are_reported(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            clean_job_ads(pd.DataFrame([{"job_title": "Analyst"}]))

    def test_dataset_preparation_selects_roles_and_removes_contact_details(self):
        long_description = (
            "Build data reports and support analytics teams using Python and SQL. "
            "This role collaborates with product partners to develop dashboards. "
            "Contact analyst@example.com or call 212-555-0198. "
            "More information is available at https://example.com/jobs."
        )
        source = pd.DataFrame(
            [
                {
                    "job_id": 1,
                    "title": "Data Analyst",
                    "description": long_description,
                },
                {
                    "job_id": 2,
                    "title": "Data Scientist / Product Analyst",
                    "description": long_description,
                },
                {
                    "job_id": 3,
                    "title": "Retail Associate",
                    "description": long_description,
                },
            ]
        )
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.csv"
            output_path = Path(directory) / "job_ads.csv"
            source.to_csv(source_path, index=False)
            prepared = prepare_dataset(source_path, output_path)

        self.assertEqual(len(prepared), 2)
        self.assertEqual(prepared["job_title"].nunique(), 2)
        self.assertIn("Data Analyst", prepared["job_title"].tolist())
        self.assertIn(
            "Data Scientist / Product Analyst", prepared["job_title"].tolist()
        )
        self.assertFalse(prepared["job_description"].str.contains("@").any())
        self.assertFalse(prepared["job_description"].str.contains("212-555-0198").any())
        self.assertFalse(prepared["job_description"].str.contains("https://").any())

    def test_pdf_export_returns_a_pdf_document(self):
        results = rank_skills(self.jobs.iloc[:2])
        pdf = build_pdf_report("Data Analyst", 2, results)
        self.assertTrue(pdf.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
