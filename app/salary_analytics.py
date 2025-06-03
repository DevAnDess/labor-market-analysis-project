import sys
import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="Software for Data Analytics Labor Market Analysis", layout="wide")

st.markdown("""
    <style>
        .stApp {
            background-color: #181818;
            color: white;
        }
        h1, h2, h3, h4, h5, h6, h7, h8 {
            color: white !important;
        }
        div[data-testid="stRadio"] label, 
        div[data-testid="stRadio"] > div > label > div {
            color: white !important;
        }        
        section[data-testid="stSidebar"] * {
            color: black !important;
        }
        .sidebar-header-black {
            color: black !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            margin-bottom: 0.5rem;
        }
        input, select, textarea, .stMultiSelect div, .stSlider, .stSelectbox {
            color: white !important;
        }
        button {
            color: black !important;
        }
        section[data-testid="stSidebar"] .stMultiSelect div[role="button"],
        section[data-testid="stSidebar"] .stMultiSelect input,
        section[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="tag"] {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='color: white;'>Software for Data Analytics Labor Market Analysis</h1>", unsafe_allow_html=True)

view_option = st.radio("Choose data source:", ("HeadHunter (Russia, Active)", "Kaggle (Global, Historical)"))

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

if view_option != st.session_state['last_data_source']:
    st.session_state['filters_applied'] = False
    st.session_state['last_data_source'] = view_option

df_filtered = df[df['source'] == 'hh_api'] if view_option.startswith("HeadHunter") else df[df['source'] == 'kaggle']

with st.sidebar.form("filter_form"):
    st.markdown("<div class='sidebar-header-black'>Filter Options</div>", unsafe_allow_html=True)
    st.markdown('**(press "apply filter" at the bottom after changes)**')

    work_year_all = sorted(df_filtered['work_year'].dropna().unique())
    selected_years = st.multiselect("Work Year", work_year_all, default=work_year_all)

    search_job = st.text_input("Search Job Title")
    job_titles_all = sorted(df_filtered['job_title'].dropna().unique())
    if search_job.strip() == "":
        job_titles_filtered = job_titles_all
    else:
        job_titles_filtered = [jt for jt in job_titles_all if search_job.lower() in jt.lower()]
        if not job_titles_filtered:
            st.warning("No job titles match your search.")

    experience_levels_all = sorted(df_filtered['experience_level'].dropna().unique())
    selected_levels = st.multiselect("Experience Level", experience_levels_all, default=experience_levels_all)

    remote_ratios_all = sorted(df_filtered['remote_ratio'].dropna().unique())
    selected_remote = st.multiselect("Remote %", remote_ratios_all, default=remote_ratios_all)

    employment_types = sorted(df_filtered['employment_type'].dropna().unique())
    selected_employment = st.multiselect("Employment Type", employment_types, default=employment_types)

    search_residence = st.text_input("Search Employee Residence")
    employee_residence_all = sorted(df_filtered['employee_residence'].dropna().unique())
    if search_residence.strip() == "":
        residence_filtered = employee_residence_all
    else:
        residence_filtered = [r for r in employee_residence_all if search_residence.lower() in r.lower()]
        if not residence_filtered:
            st.warning("No employee residences match your search.")

    search_location = st.text_input("Search Company Location")
    company_location_all = sorted(df_filtered['company_location'].dropna().unique())
    if search_location.strip() == "":
        location_filtered = company_location_all
    else:
        location_filtered = [l for l in company_location_all if search_location.lower() in l.lower()]
        if not location_filtered:
            st.warning("No company locations match your search.")

    company_sizes = sorted(df_filtered['company_size'].dropna().unique())
    selected_company_size = st.multiselect("Company Size", company_sizes, default=company_sizes)

    selected_skills = []

    if 'skills' in df_filtered.columns:
        skills_expanded = (
            df_filtered['skills']
            .dropna()
            .str.replace(r"[\[\]']", "", regex=True)
            .str.split(',\s*')
            .explode()
            .dropna()
            .unique()
        )
        skills_all = sorted(skills_expanded)
        selected_skill = st.selectbox("Select Skill", skills_all) if skills_all else None
        selected_skills = [selected_skill] if selected_skill else []

    min_salary, max_salary = int(df_filtered['salary_in_usd'].min()), int(df_filtered['salary_in_usd'].max())
    salary_range = st.slider("Salary Per Year (USD)", min_value=min_salary, max_value=max_salary, value=(min_salary, max_salary))

    apply_filters = st.form_submit_button("Apply Filter")

    reset_filters = st.form_submit_button("Reset Filter")

if reset_filters:
    st.session_state['filters_applied'] = False
    filtered_data = df_filtered.copy()



if apply_filters or st.session_state['filters_applied']:
    st.session_state['filters_applied'] = True
    filtered_data = df_filtered[
        (df_filtered['work_year'].isin(selected_years)) &
        (df_filtered['experience_level'].isin(selected_levels)) &
        (df_filtered['remote_ratio'].isin(selected_remote)) &
        (df_filtered['employment_type'].isin(selected_employment)) &
        (df_filtered['company_size'].isin(selected_company_size)) &
        (df_filtered['salary_in_usd'] >= salary_range[0]) &
        (df_filtered['salary_in_usd'] <= salary_range[1])
    ]
    if selected_skills:
        filtered_data = filtered_data[
            filtered_data['skills']
            .fillna("")
            .str.replace(r"[\[\]']", "", regex=True)
            .apply(lambda x: any(skill.strip() in x.split(',') for skill in selected_skills))
        ]

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

if view_option.startswith("Kaggle"):
    st.markdown("<h3 style='color: white;'>Average Salary by Company Location (Kaggle)</h3>", unsafe_allow_html=True)
    salary_by_location = (
        df_filtered.groupby('company_location')['salary_in_usd']
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig_loc = px.bar(salary_by_location, x='company_location', y='salary_in_usd',
                     labels={'salary_in_usd': 'Avg Salary (USD)', 'company_location': 'Company Location'},
                     title="Top 10 Company Locations by Average Salary (Kaggle)")
    st.plotly_chart(fig_loc, use_container_width=True)

if view_option.startswith("HeadHunter"):
    st.markdown("<h3 style='color: white;'>Average Salary by Skill (HeadHunter)</h3>", unsafe_allow_html=True)
    if 'skills' in df_filtered.columns:
        skill_df = df_filtered.copy()
        skill_df = skill_df.dropna(subset=['skills'])
        skill_df['skills'] = (
            skill_df['skills']
            .str.replace(r"[\[\]']", "", regex=True)
            .str.split(',\s*')
        )
        exploded = skill_df.explode('skills')
        salary_by_skill = (
            exploded.groupby('skills')['salary_in_usd']
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig_skill = px.bar(salary_by_skill, x='skills', y='salary_in_usd',
                           labels={'salary_in_usd': 'Avg Salary (USD)', 'skills': 'Skill'},
                           title="Top 10 Skills by Average Salary (HeadHunter)")
        st.plotly_chart(fig_skill, use_container_width=True)

st.markdown("<h3 style='color: white;'>Clustering Analysis</h3>", unsafe_allow_html=True)

with st.form("cluster_form"):
    st.markdown("<span style='color: white;'>Clustering Method</span>", unsafe_allow_html=True)
    cluster_algo = st.selectbox("", ["KMeans", "Hierarchical"])

    if view_option.startswith("HeadHunter"):
        st.markdown("<span style='color: white; font-weight: bold;'>Filters for HeadHunter:</span>", unsafe_allow_html=True)
        st.markdown("<span style='color: white;'>Experience Level</span>", unsafe_allow_html=True)
        cluster_exp = st.multiselect("", experience_levels_all, default=experience_levels_all)
        st.markdown("<span style='color: white;'>Company Size</span>", unsafe_allow_html=True)
        cluster_comp_size = st.multiselect("", company_sizes, default=company_sizes)
        st.markdown("<span style='color: white;'>Salary Range</span>", unsafe_allow_html=True)
        cluster_salary = st.slider("", min_value=min_salary, max_value=max_salary, value=(min_salary, max_salary))

    else:
        st.markdown("<span style='color: white; font-weight: bold;'>Filters for Kaggle:</span>", unsafe_allow_html=True)
        work_years = sorted(df_filtered['work_year'].dropna().unique())
        st.markdown("<span style='color: white;'>Work Year</span>", unsafe_allow_html=True)
        cluster_work_year = st.multiselect("", work_years, default=work_years)
        st.markdown("<span style='color: white;'>Experience Level</span>", unsafe_allow_html=True)
        cluster_exp = st.multiselect("", experience_levels_all, default=experience_levels_all)
        st.markdown("<span style='color: white;'>Salary Range</span>", unsafe_allow_html=True)
        cluster_salary = st.slider("", min_value=min_salary, max_value=max_salary, value=(min_salary, max_salary))

    run_cluster = st.form_submit_button("Run Clustering")


if run_cluster:
    if view_option.startswith("HeadHunter"):
        if cluster_algo == "KMeans":
            from clusterization.k_means.hh_salary import hh_salary
            hh_salary(cluster_exp, cluster_comp_size, cluster_salary)
        else:
            from clusterization.hierarchical.hh_salary_hierarchical import hh_salary_hierarchical
            hh_salary_hierarchical(cluster_exp, cluster_comp_size, cluster_salary)
    else:
        if cluster_algo == "KMeans":
            from clusterization.k_means.kaagle_salary import kaagle_salary
            kaagle_salary(cluster_work_year, cluster_exp, cluster_salary)
        else:
            from clusterization.hierarchical.kaagle_salary_hierarchical import kaagle_salary_hierarchical
            kaagle_salary_hierarchical(cluster_work_year, cluster_exp, cluster_salary)



st.markdown("<h3 style='color: white;'>Regression Model</h3>", unsafe_allow_html=True)

with st.form("regression_form"):
    model_type = st.radio("Choose model:", ("RandomForest", "Ridge", "CatBoost"))
    run_regression = st.form_submit_button("Run Regression")

if run_regression:
    df['work_year'] = pd.to_numeric(df['work_year'], errors='coerce')
    df = df.dropna(subset=['salary_in_usd', 'work_year'])
    df = df[(df['salary_in_usd'] >= 10000) & (df['salary_in_usd'] <= 300000)]

    cat_features = ['job_title', 'employee_residence', 'experience_level', 'employment_type', 'company_size']
    num_features = ['remote_ratio', 'work_year']
    X_cat = df[cat_features].fillna("Unknown")
    X_num = df[num_features].replace("", np.nan).astype(float).fillna(0)
    y = df['salary_in_usd']

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_encoded = encoder.fit_transform(X_cat)
    X = pd.DataFrame(X_cat_encoded).join(X_num.reset_index(drop=True))
    X.columns = X.columns.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "Ridge":
        model = Ridge(alpha=1.0)
    else:
        model = CatBoostRegressor(verbose=0)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5)

    st.subheader("Model Evaluation")
    st.markdown(f"Model: {model_type}")
    st.markdown(f"R² Score: {r2:.2f}")
    st.markdown(f"Mean Squared Error: {mse:,.0f}")
    st.markdown(f"Cross-Validation R²: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    pred_chart = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    fig_pred = px.scatter(pred_chart, x="Actual", y="Predicted",
                         title="Predicted vs Actual Salary",
                         labels={"Actual": "Actual Salary", "Predicted": "Predicted Salary"})
    fig_pred.add_shape(type="line", x0=y.min(), x1=y.max(), y0=y.min(), y1=y.max(), line=dict(color="red", dash="dash"))
    st.plotly_chart(fig_pred, use_container_width=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_names = list(encoder.get_feature_names_out(cat_features)) + num_features
        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
        top_imp = imp_df.sort_values(by="Importance", ascending=False).head(15)
        fig_imp = px.bar(top_imp, x="Importance", y="Feature", orientation="h",
                         title="Top Feature Importances")
        st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Average salary prediction by year")
    historical = df[df['work_year'] <= 2024]
    yearly = historical.groupby("work_year")["salary_in_usd"].mean().reset_index()
    X_year = yearly["work_year"].values.reshape(-1, 1)
    y_year = yearly["salary_in_usd"].values
    model_year = Ridge()
    model_year.fit(X_year, y_year)
    future_years = np.arange(2025, 2031).reshape(-1, 1)
    future_pred = model_year.predict(future_years)
    future_df = pd.DataFrame({"work_year": future_years.flatten(), "salary_in_usd": future_pred})
    fig2 = px.line(yearly, x="work_year", y="salary_in_usd", markers=True,
                   title="Average salary by year with prediction")
    fig2.add_scatter(x=future_df["work_year"], y=future_df["salary_in_usd"],
                     mode="lines+markers", name="Прогноз до 2030",
                     line=dict(color="red", dash="dot"))
    st.plotly_chart(fig2, use_container_width=True)
