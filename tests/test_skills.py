import unittest

import pandas as pd

from src.exports import build_pdf_report
from src.skills import filter_jobs_by_title, rank_skills


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
            ]
        )

    def test_title_search_matches_role_variants_case_insensitively(self):
        matches = filter_jobs_by_title(self.jobs, "data analyst")
        self.assertEqual(len(matches), 2)

    def test_skill_counts_are_per_job_posting_not_raw_mentions(self):
        results = rank_skills(self.jobs.iloc[:2])
        counts = results.set_index("skill")["job_count"]
        self.assertEqual(counts["Python"], 2)
        self.assertEqual(counts["SQL"], 1)
        self.assertEqual(counts["Power BI"], 1)

    def test_empty_search_and_empty_results_are_safe(self):
        self.assertTrue(filter_jobs_by_title(self.jobs, " ").empty)
        self.assertTrue(rank_skills(self.jobs.iloc[0:0]).empty)

    def test_word_boundaries_avoid_partial_skill_matches(self):
        jobs = pd.DataFrame(
            [{"job_description": "Use JavaScript for frontend development."}]
        )
        results = rank_skills(jobs)
        skills = set(results["skill"])
        self.assertIn("JavaScript", skills)
        self.assertNotIn("Java", skills)

    def test_pdf_export_returns_a_pdf_document(self):
        results = rank_skills(self.jobs.iloc[:2])
        pdf = build_pdf_report("Data Analyst", 2, results)
        self.assertTrue(pdf.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
