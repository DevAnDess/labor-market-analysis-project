import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_data(data):
    data["min_salary"] = data["salary"].apply(
        lambda x: int(x.split(" - ")[0]) if " - " in x and x.split(" - ")[0].isdigit() else None
    )

    grouped_by_area = data.groupby("area").size().reset_index(name="Количество вакансий")

    if "numeric_salary" in data.columns:
        data["numeric_salary"] = data["numeric_salary"].fillna(0).infer_objects(copy=False)

        avg_salary_by_area = data.groupby("area")["numeric_salary"].mean().reset_index(name="Средняя зарплата")
    else:
        avg_salary_by_area = pd.DataFrame(columns=["area", "Средняя зарплата"])

    top_companies = data["employer"].value_counts().head(10).reset_index()
    top_companies.columns = ["Компания", "Количество вакансий"]

    return {
        "grouped_by_area": grouped_by_area,
        "avg_salary_by_area": avg_salary_by_area,
        "top_companies": top_companies
    }


