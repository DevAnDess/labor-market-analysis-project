import os
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

def load_data(file_path):
    user = "sql7782452"
    password = "6HC3yNXWYM"
    host = "sql7.freesqldatabase.com"
    database = "sql7782452"

    engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")

    query = "SELECT * FROM combined_dataset_KT_format"

    df = pd.read_sql(query, engine)

    df = df.dropna(subset=["salary_in_usd"])
    df = df[(df["salary_in_usd"] >= 10000) & (df["salary_in_usd"] <= 400000)]
    df = df.fillna({
        "job_title": "Unknown",
        "employee_residence": "Unknown",
        "experience_level": "Unknown",
        "employment_type": "Full-time",
        "company_size": "Unknown",
        "remote_ratio": 0
    })
    return df

def train_model(df):
    X = df[["job_title", "experience_level", "employment_type", "company_size", "remote_ratio", "employee_residence"]]
    y = df["salary_in_usd"]

    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'),
         ["job_title", "experience_level", "employment_type", "company_size", "employee_residence"])
    ], remainder='passthrough')

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    return pipeline

base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "..", "src", "data", "processed", "combined_dataset_KT_format.csv")

df = load_data(data_path)
model = train_model(df)

trained_residences = sorted(df["employee_residence"].unique())

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("📊 Salary Prediction Dashboard"),

    html.Label("Job Title"),
    dcc.Dropdown(sorted(df["job_title"].unique()), id="job-title", value="Data Scientist"),

    html.Label("Experience Level"),
    dcc.Dropdown(sorted(df["experience_level"].unique()), id="exp-level", value="Mid"),

    html.Label("Employee Residence"),
    dcc.Dropdown(trained_residences, id="emp-res", value=trained_residences[0]),

    html.Label("Employment Type"),
    dcc.Dropdown(sorted(df["employment_type"].unique()), id="emp-type", value="Full-time"),

    html.Label("Company Size"),
    dcc.Dropdown(sorted(df["company_size"].unique()), id="comp-size", value="M"),

    html.Label("Remote Ratio"),
    dcc.Slider(min=0, max=100, step=10, value=50, id="remote-slider"),

    html.Br(),
    html.Div(id="prediction-output", style={"fontSize": 20, "fontWeight": "bold"}),

    dcc.Graph(id="salary-distribution")
])

@app.callback(
    Output("prediction-output", "children"),
    Output("salary-distribution", "figure"),
    Input("job-title", "value"),
    Input("exp-level", "value"),
    Input("emp-type", "value"),
    Input("comp-size", "value"),
    Input("remote-slider", "value"),
    Input("emp-res", "value")
)
def update_prediction(job, exp, emp_type, comp_size, remote, emp_res):
    try:
        sample = pd.DataFrame([{
            "job_title": job,
            "experience_level": exp,
            "employment_type": emp_type,
            "company_size": comp_size,
            "remote_ratio": remote,
            "employee_residence": emp_res
        }])

        predicted = model.predict(sample)[0]
        text = f"💰 Predicted Salary: ${predicted:,.0f}"

        fig = px.histogram(df, x="salary_in_usd", nbins=40, title="Salary Distribution")
        fig.add_vline(x=predicted, line_color="red", line_dash="dash")
        fig.update_layout(xaxis_title="Salary (USD)", yaxis_title="Count")

        return text, fig

    except Exception as e:
        return f"⚠️ Ошибка при предсказании: {str(e)}", px.histogram()

if __name__ == '__main__':
    app.run(debug=True)
