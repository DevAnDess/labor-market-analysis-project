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


def plot_analysis(results):
    sns.set_style("whitegrid")
    if "avg_salary_by_area" in results and not results["avg_salary_by_area"].empty:
        plt.figure(figsize=(12, 6))
        sns.histplot(results["avg_salary_by_area"]["Средняя зарплата"], bins=20, kde=True, color="blue")
        plt.title("Распределение средней зарплаты по регионам", fontsize=14, fontweight='bold')
        plt.xlabel("Средняя зарплата", fontsize=12)
        plt.ylabel("Частота", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()


    plt.figure(figsize=(12, 6))
    sns.barplot(
        y=results["top_companies"]["Компания"],
        x=results["top_companies"]["Количество вакансий"],
        hue=results["top_companies"]["Компания"],
        palette="viridis",
        legend=False
    )
    plt.title("ТОП-10 компаний по количеству вакансий", fontsize=14, fontweight='bold')
    plt.xlabel("Количество вакансий", fontsize=12)
    plt.ylabel("Компания", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()


    if "avg_salary_by_area" in results and not results["avg_salary_by_area"].empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            y=results["avg_salary_by_area"]["area"],
            x=results["avg_salary_by_area"]["Средняя зарплата"],
            hue=results["avg_salary_by_area"]["area"],
            palette="coolwarm",
            legend=False
        )
        plt.title("Средняя зарплата по регионам", fontsize=14, fontweight='bold')
        plt.xlabel("Средняя зарплата", fontsize=12)
        plt.ylabel("Регион", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.show()