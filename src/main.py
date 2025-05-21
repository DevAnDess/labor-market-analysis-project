from src.data_collection import fetch_and_combine
from src.data_processing import process_data
from src.data_analysis import analyze_data
from src.plot_analysis import plot_analysis

def main():
    combined = fetch_and_combine(
        query="аналитик данных",
        area=1,
        csv_filename="data-science-job-salaries.csv"
    )
    combined = combined[
        combined["job_title"].str.contains("аналитик|data|analyst|BI", case=False, na=False)
    ]


    print("=== Первые 5 строк объединённого датасета ===")
    print(combined.head(), "\n")

    print("=== Первые 5 строк с source = 'hh_api' ===")
    print(combined[combined["source"] == "hh_api"].head(), "\n")

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


    #plot_analysis(analysis_results)

if __name__ == "__main__":
    main()
