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
- A curated skill dictionary matches skill names and aliases in each description.
- Each skill is counted at most once per posting, then ranked by the number and share of matching postings that mention it.
- Explore the bar chart and table, then download the displayed results as CSV or PDF.

The included `data/job_ads.csv` contains manually authored **synthetic demo descriptions**. It is intended to demonstrate the analysis workflow, not to represent real job ads or current labour-market demand. Replace it with a suitably licensed, sourced dataset before drawing real-world conclusions.

## Dataset format

The CSV must contain:

| Column | Description |
| --- | --- |
| `job_title` | Role title used to filter the postings |
| `job_description` | Text searched for known skills |
| `source_type` | Provenance label for the record |

## Run tests

```powershell
python -m unittest discover -s tests -v
```
