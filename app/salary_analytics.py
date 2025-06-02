import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="Software for Data Analytics Labor Market Analysis", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #181818;
            color: #cccccc;
        }
        .stApp {
            background-color: #181818;
        }
        h1 {
            color: #fff !important;
        }
        .sidebar-header-black {
            color: black !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color: white;'>Software for Data Analytics Labor Market Analysis</h1>", unsafe_allow_html=True)

user = "sql7782452"
password = "6HC3yNXWYM"
host = "sql7.freesqldatabase.com"
database = "sql7782452"
engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")

query = "SELECT * FROM combined_dataset_KT_format"
df = pd.read_sql(query, engine)

if 'filters_applied' not in st.session_state:
    st.session_state['filters_applied'] = False
if 'last_data_source' not in st.session_state:
    st.session_state['last_data_source'] = ""
if 'last_select_all_states' not in st.session_state:
    st.session_state['last_select_all_states'] = {"job": True, "res": True, "loc": True}

st.sidebar.markdown('**(press "apply filter" at the bottom after changes)**')
st.sidebar.markdown("<div class='sidebar-header-black'>Select Dataset</div>", unsafe_allow_html=True)
view_option = st.sidebar.radio("Choose data source:", ("HeadHunter (Russia, Active)", "Kaggle (Global, Historical)"))

if view_option != st.session_state['last_data_source']:
    st.session_state['filters_applied'] = False
    st.session_state['last_data_source'] = view_option

df_filtered = df[df['source'] == 'hh_api'] if view_option.startswith("HeadHunter") else df[df['source'] == 'kaggle']

select_all_job = st.sidebar.checkbox("Select All Job Titles", value=True)
select_all_res = st.sidebar.checkbox("Select All Employee Residence", value=True)
select_all_loc = st.sidebar.checkbox("Select All Company Location", value=True)

if (
    st.session_state['last_select_all_states']["job"] != select_all_job or
    st.session_state['last_select_all_states']["res"] != select_all_res or
    st.session_state['last_select_all_states']["loc"] != select_all_loc
):
    st.session_state['filters_applied'] = True

st.session_state['last_select_all_states'] = {
    "job": select_all_job,
    "res": select_all_res,
    "loc": select_all_loc
}

with st.sidebar.form("filter_form"):
    st.markdown("<div class='sidebar-header-black'>Filter Options</div>", unsafe_allow_html=True)

    search_job = st.text_input("Search Job Title")
    job_titles_all = sorted(df_filtered['job_title'].dropna().unique())
    job_titles_filtered = [jt for jt in job_titles_all if search_job.lower() in jt.lower()]
    if select_all_job:
        selected_jobs = job_titles_filtered
    else:
        selected_jobs = st.multiselect("Job Title", job_titles_filtered, default=job_titles_filtered[:5])

    experience_levels_all = sorted(df_filtered['experience_level'].dropna().unique())
    selected_levels = st.multiselect("Experience Level", experience_levels_all, default=experience_levels_all)

    remote_ratios_all = sorted(df_filtered['remote_ratio'].dropna().unique())
    selected_remote = st.multiselect("Remote %", remote_ratios_all, default=remote_ratios_all)

    employment_types = sorted(df_filtered['employment_type'].dropna().unique())
    selected_employment = st.multiselect("Employment Type", employment_types, default=employment_types)

    search_residence = st.text_input("Search Employee Residence")
    employee_residence_all = sorted(df_filtered['employee_residence'].dropna().unique())
    residence_filtered = [r for r in employee_residence_all if search_residence.lower() in r.lower()]
    if select_all_res:
        selected_residence = residence_filtered
    else:
        selected_residence = st.multiselect("Employee Residence", residence_filtered, default=residence_filtered[:5])

    search_location = st.text_input("Search Company Location")
    company_location_all = sorted(df_filtered['company_location'].dropna().unique())
    location_filtered = [l for l in company_location_all if search_location.lower() in l.lower()]
    if select_all_loc:
        selected_company_location = location_filtered
    else:
        selected_company_location = st.multiselect("Company Location", location_filtered, default=location_filtered[:5])

    company_sizes = sorted(df_filtered['company_size'].dropna().unique())
    selected_company_size = st.multiselect("Company Size", company_sizes, default=company_sizes)

    min_salary, max_salary = int(df_filtered['salary_in_usd'].min()), int(df_filtered['salary_in_usd'].max())
    salary_range = st.slider("Salary Per Year (USD)", min_value=min_salary, max_value=max_salary, value=(min_salary, max_salary))
    sort_column = st.selectbox("Sort by", df_filtered.columns)
    sort_asc = st.checkbox("Sort ascending?", value=False)

    apply_filters = st.form_submit_button("Apply Filter")

if apply_filters or st.session_state['filters_applied']:
    st.session_state['filters_applied'] = True
    filtered_data = df_filtered[
        (df_filtered['job_title'].isin(selected_jobs)) &
        (df_filtered['experience_level'].isin(selected_levels)) &
        (df_filtered['remote_ratio'].isin(selected_remote)) &
        (df_filtered['employment_type'].isin(selected_employment)) &
        (df_filtered['employee_residence'].isin(selected_residence)) &
        (df_filtered['company_location'].isin(selected_company_location)) &
        (df_filtered['company_size'].isin(selected_company_size)) &
        (df_filtered['salary_in_usd'] >= salary_range[0]) &
        (df_filtered['salary_in_usd'] <= salary_range[1])
    ]
    filtered_data = filtered_data.sort_values(by=sort_column, ascending=sort_asc)
else:
    filtered_data = df_filtered.copy()

if 'filtered_data' not in locals():
    filtered_data = df_filtered.copy()

if 'requirement' in filtered_data.columns:
    filtered_data['requirement'] = filtered_data['requirement'].str.replace(r'</?highlighttext>', '', regex=True)

st.markdown("<h3 style='color: white;'>Filtered Job Listings</h3>", unsafe_allow_html=True)
st.dataframe(filtered_data, use_container_width=True)

st.markdown("<h3 style='color: white;'>Average Salary by Job Title</h3>", unsafe_allow_html=True)
salary_chart = (
    df_filtered.groupby('job_title')['salary_in_usd']
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig = px.bar(salary_chart, x='job_title', y='salary_in_usd',
             labels={'salary_in_usd': 'Avg Salary (USD)', 'job_title': 'Job Title'},
             title="Top 10 Job Titles by Average Salary")
st.plotly_chart(fig, use_container_width=True)

st.markdown("<h3 style='color: white;'>Clustering Analysis</h3>", unsafe_allow_html=True)

cluster_algo = st.selectbox("Clustering Method", ["KMeans", "Hierarchical"])
if view_option.startswith("HeadHunter"):
    st.markdown("**Filters for HeadHunter:**")
    cluster_exp = st.multiselect("Experience Level", experience_levels_all, default=experience_levels_all)
    cluster_comp_size = st.multiselect("Company Size", company_sizes, default=company_sizes)
    cluster_salary = st.slider("Salary Range", min_value=min_salary, max_value=max_salary, value=(min_salary, max_salary))

    run_cluster = st.button("Run Clustering")

    if run_cluster:
        from clusterization.k_means.hh_salary import hh_salary
        from clusterization.hierarchical.hh_salary_hierarchical import hh_salary_hierarchical
        if cluster_algo == "KMeans":
            hh_salary(cluster_exp, cluster_comp_size, cluster_salary)
        else:
            hh_salary_hierarchical(cluster_exp, cluster_comp_size, cluster_salary)

else:
    st.markdown("**Filters for Kaggle:**")
    work_years = sorted(df_filtered['work_year'].dropna().unique())
    cluster_work_year = st.multiselect("Work Year", work_years, default=work_years)
    cluster_exp = st.multiselect("Experience Level", experience_levels_all, default=experience_levels_all)
    cluster_salary = st.slider("Salary Range", min_value=min_salary, max_value=max_salary, value=(min_salary, max_salary))

    run_cluster = st.button("Run Clustering")

    if run_cluster:
        from clusterization.k_means.kaagle_salary import kaagle_salary
        from clusterization.hierarchical.kaagle_salary_hierarchical import kaagle_salary_hierarchical
        if cluster_algo == "KMeans":
            kaagle_salary(cluster_work_year, cluster_exp, cluster_salary)
        else:
            kaagle_salary_hierarchical(cluster_work_year, cluster_exp, cluster_salary)



