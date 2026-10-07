from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.exports import build_pdf_report
from src.skills import clean_job_ads, filter_jobs_by_title, rank_skills

DATA_PATH = Path(__file__).parent / "data" / "job_ads.csv"

st.set_page_config(
    page_title="Tech Skills Recommender",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_job_ads(path: str) -> pd.DataFrame:
    return clean_job_ads(pd.read_csv(path))


st.title("Tech Skills Recommender")
st.write(
    "Explore the technical skills most often mentioned in job descriptions "
    "for a role you're interested in."
)

try:
    job_ads = load_job_ads(str(DATA_PATH))
except FileNotFoundError:
    st.error(f"Dataset not found at `{DATA_PATH}`. Restore the project data file to continue.")
    st.stop()
except pd.errors.ParserError as error:
    st.error(f"The dataset could not be read as a valid CSV: {error}")
    st.stop()
except ValueError as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.metric("Job descriptions in dataset", len(job_ads))
    st.caption(
        "This role-focused sample contains real postings with scrape timestamps "
        "mainly from November 2023. It is historical data, not a live job feed "
        "or a measure of current demand."
    )
    st.divider()
    st.markdown(
        "**How skills are identified**  \n"
        "NLTK tokenizes descriptions; recognized technical skill names and aliases "
        "are counted at most once per posting. Generic phrases are excluded."
    )
    st.markdown(
        "**Data source**  \n"
        "[xanderios/linkedin-job-postings](https://huggingface.co/datasets/"
        "xanderios/linkedin-job-postings), labeled MIT."
    )

query = st.text_input(
    "Enter a job title",
    placeholder="e.g. Data Analyst or Software Engineer",
)

if query.strip():
    matching_jobs = filter_jobs_by_title(job_ads, query)
    if matching_jobs.empty:
        st.warning(
            "No matching job titles were found. Try Data Analyst, Data Scientist, "
            "Data Engineer, Software Engineer, Software Developer, or Product Analyst."
        )
    else:
        skill_results = rank_skills(matching_jobs)
        if skill_results.empty:
            st.info(
                "No recognized technical skills were found. The skill catalog may "
                "need additional terms for this role."
            )
        else:
            st.subheader(f"Technical skills for “{query.strip()}”")
            st.caption(
                "Skills are identified using a maintained technical-skill vocabulary "
                "and NLTK phrase matching. Unlisted skills will not appear; review "
                "the source postings before treating counts as recommendations."
            )
            first, second = st.columns(2)
            first.metric("Matching job descriptions", len(matching_jobs))
            second.metric("Technical skills found", len(skill_results))

            top_n = st.slider(
                "Number of skills to display",
                min_value=1,
                max_value=min(20, len(skill_results)),
                value=min(5, len(skill_results)),
            )
            visible_skills = skill_results.head(top_n)
            chart = px.bar(
                visible_skills.sort_values("job_count"),
                x="job_count",
                y="skill",
                orientation="h",
                labels={
                    "job_count": "Job descriptions mentioning skill",
                    "skill": "Technical skill",
                },
                text="job_count",
                color_discrete_sequence=["#3478F6"],
            )
            chart.update_layout(
                height=max(350, 34 * len(visible_skills)),
                showlegend=False,
                yaxis_title=None,
                xaxis_title="Number of matching job descriptions",
                margin=dict(l=8, r=16, t=16, b=8),
            )
            chart.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(chart, width="stretch")

            display_results = visible_skills.rename(
                columns={
                    "skill": "Technical skill",
                    "job_count": "Job descriptions",
                    "share_percent": "Share of matching jobs (%)",
                }
            ).copy()
            display_results["Share of matching jobs (%)"] = (
                display_results["Share of matching jobs (%)"].round(1)
            )
            st.dataframe(display_results, hide_index=True, width="stretch")

            csv_export = visible_skills.assign(
                source=matching_jobs["source_type"].iloc[0],
                source_url="https://huggingface.co/datasets/xanderios/linkedin-job-postings",
            )
            csv_data = csv_export.to_csv(index=False).encode("utf-8-sig")
            pdf_data = build_pdf_report(query.strip(), len(matching_jobs), visible_skills)
            csv_column, pdf_column = st.columns(2)
            csv_column.download_button(
                "Download results as CSV",
                data=csv_data,
                file_name="recommended_skills.csv",
                mime="text/csv",
                width="stretch",
            )
            pdf_column.download_button(
                "Download results as PDF",
                data=pdf_data,
                file_name="recommended_skills.pdf",
                mime="application/pdf",
                width="stretch",
            )

            with st.expander("View matching job descriptions"):
                st.dataframe(
                    matching_jobs[["job_title", "job_description"]],
                    hide_index=True,
                    width="stretch",
                )
else:
    st.info("Enter a job title to explore the sample descriptions and skill recommendations.")
