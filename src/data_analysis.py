import pandas as pd


def extract_min_salary(salary):
    if isinstance(salary, str) and " - " in salary:
        min_salary = salary.split(" - ")[0]
        return int(min_salary) if min_salary.isdigit() else None
    return None


def analyze_data(data):
    data["min_salary"] = data["salary"].apply(extract_min_salary)
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
