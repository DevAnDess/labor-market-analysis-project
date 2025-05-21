import matplotlib.pyplot as plt
import seaborn as sns

def plot_analysis(results):

    if "avg_salary_by_area" in results and not results["avg_salary_by_area"].empty:
        plt.figure(figsize=(12, 6))
        sns.set_style("darkgrid")
        sns.histplot(
            results["avg_salary_by_area"]["Средняя зарплата"],
            bins=20, kde=True, color="royalblue", alpha=0.6
        )
        plt.axvline(results["avg_salary_by_area"]["Средняя зарплата"].mean(), color='red', linestyle='--',
                    label="Средняя зарплата")
        plt.axvline(results["avg_salary_by_area"]["Средняя зарплата"].median(), color='green', linestyle='--',
                    label="Медианная зарплата")
        plt.title("Распределение средней зарплаты по регионам", fontsize=16, fontweight='bold', color='darkblue')
        plt.xlabel("Средняя зарплата", fontsize=14)
        plt.ylabel("Частота", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()


    if "top_companies" in results and not results["top_companies"].empty:
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
            y=results["avg_salary_by_area"]["Регион"],
            x=results["avg_salary_by_area"]["Средняя зарплата"],
            hue=results["avg_salary_by_area"]["Регион"],
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


    if "top_companies" in results and not results["top_companies"].empty:
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        plt.pie(
            results["top_companies"]["Количество вакансий"],
            labels=results["top_companies"]["Компания"],
            autopct='%1.1f%%',
            colors=sns.color_palette("pastel"),
            startangle=140,
            wedgeprops={'edgecolor': 'black'}
        )
        plt.title(f"Распределение вакансий среди ТОП-{len(results['top_companies'])} компаний", fontsize=14,
                  fontweight='bold')
        plt.show()
