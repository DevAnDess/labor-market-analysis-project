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

    print("\nГруппировка по регионам:")
    print(grouped_by_area)

    print("\nСредняя зарплата по регионам:")
    print(avg_salary_by_area)

    print("\nТОП-10 компаний по количеству вакансий:")
    print(top_companies)

    return {
        "grouped_by_area": grouped_by_area,
        "avg_salary_by_area": avg_salary_by_area,
        "top_companies": top_companies
    }


def plot_analysis(results):
    sns.set_style("whitegrid")

    plt.figure(figsize=(12, 6))
    sns.barplot(
        y=results["grouped_by_area"]["area"],
        x=results["grouped_by_area"]["Количество вакансий"],
        hue=results["grouped_by_area"]["area"],
        palette="Blues_r",
        legend=False
    )
    plt.title("Количество вакансий по регионам", fontsize=14, fontweight='bold')
    plt.xlabel("Количество вакансий", fontsize=12)
    plt.ylabel("Регион", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
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
