# Project Instructions: Data App (Tech Skills Recommender)

## Project goal
Build a Python-powered data app that recommends top technical skills based on a job title. The app should simulate using data analysis to help students identify which skills to prioritise when applying for jobs.

## Core functionality
- Build a Streamlit app that allows users to type in a job title (for example: Data Analyst, Software Engineer).
- Use a CSV dataset of job descriptions (can be downloaded or created manually).
- Extract and visualise the top recurring skills for that title using Python and Pandas.
- Display results visually using bar charts or word clouds.
- Allow users to download results as a CSV or PDF.

## Current technical stack
- Python
- Streamlit
- Pandas
- NLTK for tokenizing descriptions and extracting recurring one-, two-, and three-word keywords without a fixed skills dictionary
- Plotly for visualisations
- ReportLab for PDF exports

## Implementation expectations
- Keep the app simple, polished, and easy to run locally.
- Store the demo data in `data/job_ads.csv`; document its schema and provenance.
- Use Pandas to validate, clean, and preprocess CSV text data.
- Use NLTK to tokenize descriptions and extract recurring keywords (candidate skills), rather than relying only on a hand-maintained skill dictionary.
- Rank keywords by the number of matching job postings that mention them; count each keyword at most once per posting.
- Present the output in a clear, user-friendly dashboard.
- Keep code organised into modules if the app grows beyond a single file.
- Clearly distinguish synthetic examples from real job-market evidence.
- Document how to deploy the app to Streamlit Community Cloud from the GitHub repository.

## Suggested project structure
- `app.py` or `main.py` for the Streamlit entry point
- `data/` for CSV datasets
- `src/` for skill analysis and export helpers
- `tests/` for automated tests
- `requirements.txt` for Python dependencies
- `README.md` for run instructions and dataset documentation

## Quality bar
- Code should be readable and well-structured.
- Use clear variable names and comments only where needed.
- Prefer reproducible logic over hardcoded values.
- Validate that the app runs successfully with a local Python environment.
- Ensure the UI is responsive and the charts are understandable.

## Delivery standard
This project should be implemented as a working demo app that demonstrates the full flow:
1. User enters a job title.
2. Data is loaded and processed.
3. Relevant skills are extracted and summarised.
4. Results are displayed in charts and/or word clouds.
5. Results can be exported.
6. The app can be deployed from the GitHub repository using Streamlit Community Cloud.

## Notes for future work
- Add support for multiple job titles or multiple datasets.
- Review extracted keywords to distinguish technical skills from other recurring terms.
- Add filters for seniority, experience level, or industry.
- Deploy the app on Streamlit Community Cloud.
