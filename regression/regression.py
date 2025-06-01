import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sqlalchemy import create_engine


def load_data(path, title_filter=None):
    user = "sql7782452"
    password = "6HC3yNXWYM"
    host = "sql7.freesqldatabase.com"
    database = "sql7782452"

    engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")

    query = "SELECT * FROM combined_dataset_KT_format"

    df = pd.read_sql(query, engine)

    df = df.dropna(subset=["salary_in_usd"])
    df["Salary"] = pd.to_numeric(df["salary_in_usd"], errors="coerce")
    df = df[(df["Salary"] >= 10000) & (df["Salary"] <= 300000)]

    if title_filter:
        df = df[df["job_title"].str.lower().str.contains(title_filter.lower())]

    df = df.fillna({
        "job_title": "Unknown",
        "employee_residence": "Unknown",
        "experience_level": "Unknown",
        "employment_type": "Full-time",
        "company_size": "Unknown",
        "remote_ratio": 0
    })

    df["title_country"] = df["job_title"] + "_" + df["employee_residence"]
    df["title_year"] = df["job_title"] + "_" + df.get("work_year", 0).astype(str)

    mean_salary_by_group = df.groupby("title_country")["Salary"].mean()
    df["avg_salary_title_loc"] = df["title_country"].map(mean_salary_by_group)

    mean_salary_by_year = df.groupby("title_year")["Salary"].mean()
    df["avg_salary_title_year"] = df["title_year"].map(mean_salary_by_year)

    return df


def preprocess_data(df):
    top_jobs = df["job_title"].value_counts().nlargest(20).index
    df["job_title"] = df["job_title"].apply(lambda x: x if x in top_jobs else "Other")

    cat_features = ["job_title", "employee_residence", "experience_level", "employment_type", "company_size"]
    num_features = ["remote_ratio", "avg_salary_title_loc", "avg_salary_title_year"]

    if "work_year" in df.columns:
        num_features.append("work_year")

    X_cat = df[cat_features]
    X_num = df[num_features]
    y_log = np.log1p(df["Salary"])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_encoded = encoder.fit_transform(X_cat)
    X = pd.DataFrame(X_cat_encoded).join(X_num.reset_index(drop=True))
    X.columns = X.columns.astype(str)

    return X, y_log, encoder, cat_features + num_features, df


def evaluate_model(model, X_test, y_test_log):
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test = np.expm1(y_test_log)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_test, y_pred


def cross_validate(model, X, y_log, name=""):
    scores = cross_val_score(model, X, y_log, cv=5, scoring="r2")
    print(f"\n {name} Cross-Val R²: {scores.mean():.2f} ± {scores.std():.2f}")


def plot_predictions(y_test, y_pred, model_name, base_dir):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"{model_name}: Predicted vs Actual Salary")
    plt.xlabel("Actual Salary (USD)")
    plt.ylabel("Predicted Salary (USD)")
    plt.grid(True)
    plt.tight_layout()

    plots_dir = os.path.join(base_dir, "plots_regression")
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, f"{model_name}_plot.png")
    plt.savefig(save_path)
    print(f" График сохранён: {save_path}")
    plt.show()


def show_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[-20:]
        plt.figure(figsize=(8, 6))
        plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()
        plt.show()


def analyze_errors(y_test, y_pred, df_test, base_dir):
    errors = np.abs(y_test - y_pred)
    df_test = df_test.copy()
    df_test["Error"] = errors
    top_errors = df_test.sort_values("Error", ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    plt.barh(top_errors["job_title"] + " / " + top_errors["employee_residence"], top_errors["Error"])
    plt.xlabel("Prediction Error (USD)")
    plt.title("Top 20 Errors by Title and Location")
    plt.tight_layout()
    error_path = os.path.join(base_dir, "plots_regression", "TopErrors.png")
    plt.savefig(error_path)
    print(f" Анализ ошибок сохранён: {error_path}")
    plt.show()


def forecast_salary_by_year(df, base_dir):
    if "work_year" in df.columns:
        years_range = range(df["work_year"].min(), 2025)
        df_year = df.groupby("work_year")["Salary"].mean().reindex(years_range).interpolate().reset_index()
        df_year.columns = ["work_year", "Salary"]

        X_year = df_year["work_year"].values.reshape(-1, 1)
        y_year = df_year["Salary"].values

        model = Ridge()
        model.fit(X_year, y_year)

        future_years = np.arange(2025, 2031).reshape(-1, 1)
        future_salaries = model.predict(future_years)

        plt.figure(figsize=(10, 6))
        plt.plot(df_year["work_year"], df_year["Salary"], marker='o', label="Исторические данные")
        plt.plot(future_years.flatten(), future_salaries, marker='x', linestyle='--', label="Прогноз до 2030")
        plt.xlabel("Год")
        plt.ylabel("Средняя зарплата (USD)")
        plt.title("Прогноз средней зарплаты Data Scientist по годам")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plots_dir = os.path.join(base_dir, "plots_regression")
        os.makedirs(plots_dir, exist_ok=True)
        forecast_path = os.path.join(plots_dir, "SalaryForecastTo2030.png")
        plt.savefig(forecast_path)
        print(f" Прогноз сохранён: {forecast_path}")
        plt.show()


def main():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "src", "data", "processed", "combined_dataset_KT_format.csv")
    df = load_data(data_path)

    X, y_log, encoder, feature_names, df_ready = preprocess_data(df)
    X_train, X_test, y_train_log, y_test_log, df_train, df_test = train_test_split(
        X, y_log, df_ready, test_size=0.2, random_state=42
    )

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=None, min_samples_split=5, random_state=42),
        "Ridge": Ridge(alpha=1.0),
        "CatBoost": CatBoostRegressor(verbose=0, iterations=200, learning_rate=0.1, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train_log)
        mse, r2, y_test, y_pred = evaluate_model(model, X_test, y_test_log)
        print(f"\n {name} Results:")
        print(f"  RMSE: {np.sqrt(mse):,.2f} USD | R²: {r2:.2f}")
        cross_validate(model, X, y_log, name)
        plot_predictions(y_test, y_pred, name, base_dir)
        show_feature_importance(model, X.columns)
        analyze_errors(y_test, y_pred, df_test, base_dir)

    forecast_salary_by_year(df, base_dir)


if __name__ == "__main__":
    main()
