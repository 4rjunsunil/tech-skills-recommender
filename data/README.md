# Job-postings dataset

`job_ads.csv` is a prepared subset of the public [LinkedIn Job Postings dataset](https://huggingface.co/datasets/xanderios/linkedin-job-postings), published by `xanderios` on Hugging Face. The dataset repository labels its data **MIT** licensed. The source file contains 33,246 postings and includes `title` and `description` fields.

The project pins the source snapshot at commit [`99206394a348ce110e192cc2349fdcd7d9719d1c`](https://huggingface.co/datasets/xanderios/linkedin-job-postings/tree/99206394a348ce110e192cc2349d7d9719d1c). The prepared subset contains 356 postings:

| Role identified from title | Postings |
| --- | ---: |
| Data Analyst | 80 |
| Data Scientist | 66 |
| Data Engineer | 80 |
| Software Engineer | 80 |
| Software Developer | 36 |
| Product Analyst | 14 |

The preparation script deterministically samples at most 80 postings per role, removes rows without a title or description, deduplicates by source job ID, and removes email addresses, URLs, and phone numbers from descriptions. Original titles and descriptions are retained for keyword analysis. The `source_type` column and this file provide dataset attribution.

## Important limitations and rights

These are historical postings, not live vacancies. Their scraped timestamps are mainly from November 2023; missing or invalid timestamps may also occur. Results show patterns in this sample and should not be presented as current labour-market demand.

The Hugging Face repository labels the dataset MIT licensed, but job descriptions may contain text originally authored by employers or other third parties. Review the source dataset's terms and any applicable rights before redistributing the prepared CSV or using it commercially. The sample is included here for this educational project with attribution.

## Regenerate the subset

1. Download `job_postings.csv` from the [source dataset page](https://huggingface.co/datasets/xanderios/linkedin-job-postings).
2. Run the preparation script with the downloaded file:

   ```powershell
   python data/prepare_job_ads.py path\to\job_postings.csv
   ```

The script writes `data/job_ads.csv` by default. It requires pandas, already listed in the project's `requirements.txt`.
