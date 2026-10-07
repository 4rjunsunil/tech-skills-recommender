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
- The app finds sample job titles containing that search.
- Pandas checks the required columns, trims text, and removes rows without a title or description.
- NLTK tokenizes each description and extracts one-, two-, and three-word keywords. Keywords are discovered from the text rather than looked up in a fixed skill dictionary, so new terms in the data can appear in the results.
- Each keyword is counted at most once per posting, then ranked by the number and share of matching postings that mention it.
- Explore the bar chart and table, then download the displayed results as CSV or PDF.

Keyword extraction can include recurring terms that are not skills. Review results in context before treating them as recommendations. The app uses NLTK's regular-expression tokenizer and does not require downloading an additional language model or corpus.

The included `data/job_ads.csv` contains manually authored **synthetic demo descriptions**. It is intended to demonstrate the analysis workflow, not to represent real job ads or current labour-market demand. Replace it with a suitably licensed, sourced dataset before drawing real-world conclusions.

## Dataset format

The CSV must contain:

| Column | Description |
| --- | --- |
| `job_title` | Role title used to filter the postings |
| `job_description` | Text processed by NLTK to extract recurring keywords |
| `source_type` | Provenance label for the record |
