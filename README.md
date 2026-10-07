# Tech Skills Recommender

A small Streamlit data app that lets students explore technical skills mentioned in job descriptions for a chosen role.

## Run locally

1. Use Python 3.10 or newer.
2. (Recommended) Create and activate a virtual environment.
3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Start the app:

   ```powershell
   streamlit run app.py
   ```

## How it works

- Enter a job title such as `Data Analyst`, `Data Scientist`, or `Software Engineer`.
- The app finds job titles in the prepared dataset containing that search.
- Pandas checks the required columns, trims text, and removes rows without a title or description.
- NLTK tokenizes each description into words and phrases. Those phrases are matched against a curated technical-skill vocabulary and its aliases, rather than treating every common phrase as a skill.
- Each recognized skill counts at most once per posting, then ranks by the number and share of matching postings that mention it.
- Explore the bar chart and table, then download the displayed results as CSV or PDF.

The vocabulary excludes generic terms such as “design” and “code,” which appeared prominently when all frequent phrases were treated as skills. A tradeoff is that a skill absent from the vocabulary will not appear until its name or alias is added in `src/skills.py`. Review results in the context of the source postings; frequency in this historical sample is not proof that a skill is essential today. The app uses NLTK's regular-expression tokenizer and does not require downloading an additional language model or corpus.

The included `data/job_ads.csv` contains a limited role-focused sample of real job postings from the public `xanderios/linkedin-job-postings` dataset on Hugging Face. The source repository labels the dataset **MIT** licensed. The postings are historical (collected in 2023), not live vacancies or evidence of current demand. See [`data/README.md`](data/README.md) for attribution, limitations, and instructions to regenerate the sample.

## Dataset format

The CSV must contain:

| Column | Description |
| --- | --- |
| `job_title` | Role title used to filter the postings |
| `job_description` | Text processed by NLTK to match recognized technical skill names and aliases |
| `source_type` | Dataset attribution and license label |

## Dataset attribution and license

The app's prepared subset is derived from [LinkedIn Job Postings](https://huggingface.co/datasets/xanderios/linkedin-job-postings), maintained on Hugging Face by `xanderios`. The source dataset repository labels its data **MIT** licensed. The project includes attribution in every row's `source_type` and in [`data/README.md`](data/README.md). The source data itself contains third-party job-posting text; review the source terms and any applicable rights before redistributing the data or using it commercially.

To regenerate the role-focused subset, download the source CSV from the dataset page, then run:

```powershell
python data/prepare_job_ads.py path\to\job_postings.csv
```
