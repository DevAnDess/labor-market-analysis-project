from src.data_collection import fetch_and_combine
from src.data_processing import infer_missing_fields_from_text
from src.data_analysis import analyze_data
from src.plot_analysis import plot_analysis


def main():
    combined = fetch_and_combine(
        query="аналитик данных",
        area=1,
        csv_filename="data-science-job-salaries.csv"
    )

    combined = infer_missing_fields_from_text(combined)

    print("=== Первые 5 строк объединённого датасета ===")
    print(combined.head(), "\n")

    print("=== Первые 5 строк с source = 'hh_api' ===")
    print(combined[combined["source"] == "hh_api"].head(15), "\n")

    print("=== Примеры описаний вакансий HH API ===")
    hh_descriptions = combined[combined["source"] == "hh_api"]["requirement"].dropna().head(5).values
    for i, desc in enumerate(hh_descriptions, 1):
        print(f"{i}) {desc[:300]}...\n")

    print("=== Распределение по source ===")
    print(combined["source"].value_counts(), "\n")

    print("=== Статистика salary_in_usd ===")
    print(combined["salary_in_usd"].describe(), "\n")

    print("=== Пропуски по колонкам ===")
    print(combined.isna().sum(), "\n")

    analysis_results = analyze_data(combined)
    for key, df in analysis_results.items():
        print(f"--- {key} ---")
        print(df.head(), "\n")

    # plot_analysis(analysis_results)


if __name__ == "__main__":
    main()
