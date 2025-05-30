import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

def load_data(path, title_filter=None):
    df = pd.read_csv(path)
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

def preprocess_data(df):
    cat_features = ["job_title", "employee_residence", "experience_level", "employment_type", "company_size"]
    num_features = ["remote_ratio"]

    if "skills" in df.columns:
        cat_features.append("skills")

    X_cat = df[cat_features]
    X_num = df[num_features]
    y_log = np.log1p(df["Salary"])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_encoded = encoder.fit_transform(X_cat)
    X = pd.DataFrame(X_cat_encoded).join(X_num.reset_index(drop=True))
    X.columns = X.columns.astype(str)

    return X, y_log, encoder, cat_features + num_features

def evaluate_model(model, X_test, y_test_log):
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test = np.expm1(y_test_log)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_test, y_pred

def cross_validate(model, X, y_log, name=""):
    scores = cross_val_score(model, X, y_log, cv=5, scoring="r2")
    print(f"📈 {name} Cross-Val R²: {scores.mean():.2f} ± {scores.std():.2f}")

def plot_predictions(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"{model_name}: Predicted vs Actual Salary")
    plt.xlabel("Actual Salary (USD)")
    plt.ylabel("Predicted Salary (USD)")
    plt.grid(True)
    plt.tight_layout()

    base_dir = os.path.dirname(__file__)
    plots_dir = os.path.join(base_dir, "plots_regression")
    os.makedirs(plots_dir, exist_ok=True)

    save_path = os.path.join(plots_dir, f"{model_name}_plot.png")
    plt.savefig(save_path)
    print(f"📊 График сохранён: {save_path}")
    plt.show()

def show_feature_importance(model, feature_names, title="Feature Importance"):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[-20:]
        plt.figure(figsize=(8, 6))
        plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
        plt.title(title)
        plt.tight_layout()
        plt.show()

def main():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "src", "data", "processed", "combined_dataset_KT_format.csv")
    df = load_data(data_path)

    X, y_log, encoder, feature_names = preprocess_data(df)
    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=150, max_depth=None, min_samples_split=5, random_state=42)
    rf.fit(X_train, y_train_log)
    mse, r2, y_test, y_pred = evaluate_model(rf, X_test, y_test_log)
    print("\n🌲 RandomForest:")
    print(f"  RMSE: {np.sqrt(mse):,.2f} USD | R²: {r2:.2f}")
    cross_validate(rf, X, y_log, "RandomForest")
    plot_predictions(y_test, y_pred, "RandomForest")
    show_feature_importance(rf, X.columns, "RandomForest Feature Importance")

    lr = LinearRegression()
    lr.fit(X_train, y_train_log)
    mse, r2, y_test, y_pred = evaluate_model(lr, X_test, y_test_log)
    print("\n📉 LinearRegression:")
    print(f"  RMSE: {np.sqrt(mse):,.2f} USD | R²: {r2:.2f}")
    cross_validate(lr, X, y_log, "LinearRegression")
    plot_predictions(y_test, y_pred, "LinearRegression")

    cat_model = CatBoostRegressor(verbose=0, iterations=200, learning_rate=0.1, random_state=42)
    cat_model.fit(X_train, y_train_log)
    mse, r2, y_test, y_pred = evaluate_model(cat_model, X_test, y_test_log)
    print("\n🐱 CatBoost:")
    print(f"  RMSE: {np.sqrt(mse):,.2f} USD | R²: {r2:.2f}")
    cross_validate(cat_model, X, y_log, "CatBoost")
    plot_predictions(y_test, y_pred, "CatBoost")
    show_feature_importance(cat_model, X.columns, "CatBoost Feature Importance")

if __name__ == "__main__":
    main()
