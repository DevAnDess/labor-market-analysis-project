import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine


def load_data(file_path: str, title_filter: str = None) -> pd.DataFrame:
    user = "sql7782452"
    password = "6HC3yNXWYM"
    host = "sql7.freesqldatabase.com"
    database = "sql7782452"

    engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")

    query = "SELECT * FROM combined_dataset_KT_format"

    df = pd.read_sql(query, engine)

    df = df[df["source"].str.lower() == "kaggle"]
    df = df.dropna(subset=["salary_in_usd"])
    df["Salary"] = pd.to_numeric(df["salary_in_usd"], errors="coerce")
    df = df[(df["Salary"] >= 10000) & (df["Salary"] <= 400000)]

    if title_filter:
        df = df[df["job_title"].str.lower().str.contains(title_filter.lower())]

    df = df.fillna({
        "job_title": "Unknown",
        "employee_residence": "Unknown",
        "experience_level": "Unknown",
        "employment_type": "Full-time",
        "company_size": "Unknown",
        "remote_ratio": 0,
        "skills": ""
    })

    return df


def preprocess_data(df: pd.DataFrame):
    features_cat = [
        "job_title", "employee_residence", "experience_level",
        "employment_type", "company_size"
    ]

    if "skills" in df.columns:
        features_cat.append("skills")

    features_num = ["remote_ratio"]

    X_cat = df[features_cat]
    X_num = df[features_num]
    y = df["Salary"]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_encoded = encoder.fit_transform(X_cat)

    X = pd.DataFrame(X_cat_encoded).join(X_num.reset_index(drop=True))
    X.columns = X.columns.astype(str)

    return X, y, encoder


def train_best_model(X, y):
    rf = RandomForestRegressor(random_state=42)

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    }

    grid = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
    grid.fit(X, y)

    print(f" Best params from GridSearchCV: {grid.best_params_}")
    return grid.best_estimator_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred


def plot_predictions(y_test, y_pred, output_path="salary_plot.png"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Predicted vs Actual Salary")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f" График сохранён в: {output_path}")
    plt.show()


def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f" Cross-Validation R² Scores: {scores}")
    print(f" Mean R²: {scores.mean():.2f} | Std: {scores.std():.2f}")


def main():
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, "..", "src", "data", "processed", "combined_dataset_KT_format.csv")
    dataset_path = os.path.abspath(dataset_path)

    df = load_data(dataset_path, title_filter=None)
    X, y, encoder = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_best_model(X_train, y_train)

    mse, r2, y_pred = evaluate_model(model, X_test, y_test)
    print(" Salary Prediction (Improved Final Model)")
    print(f" Mean Squared Error (MSE): {mse:,.2f}")
    print(f" R² Score: {r2:.2f}")

    sample = pd.DataFrame([{
        "job_title": "Data Scientist",
        "employee_residence": "Germany",
        "experience_level": "Mid",
        "employment_type": "Full-time",
        "company_size": "M",
        "remote_ratio": 50,
        "skills": "python sql machine learning"
    }])
    sample_cat = sample[
        ["job_title", "employee_residence", "experience_level", "employment_type", "company_size", "skills"]]
    sample_num = sample[["remote_ratio"]]
    sample_encoded = encoder.transform(sample_cat)
    sample_full = pd.DataFrame(sample_encoded).join(sample_num)
    sample_full.columns = sample_full.columns.astype(str)

    predicted_salary = model.predict(sample_full)[0]
    print(f"Predicted salary for sample: ${predicted_salary:,.2f}")

    print("Real vs Predicted:")
    for i in range(min(5, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        print(f"  ▶ Actual: ${actual:,.2f} | Predicted: ${predicted:,.2f}")

    plot_predictions(y_test, y_pred)

    cross_validate_model(model, X, y)


if __name__ == "__main__":
    main()
