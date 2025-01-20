from src.data_collection import fetch_all_vacancies
from src.data_processing import process_data
from src.data_analysis import analyze_data
from src.plot_analysis import plot_analysis


def main():
    raw_data = fetch_all_vacancies()

    clean_data = process_data(raw_data)

    analysis_results = analyze_data(clean_data)

    print(clean_data.head())

    for key, value in analysis_results.items():
        print(f"\n{key}:")
        print(value.head())

    plot_analysis(analysis_results)


if __name__ == "__main__":
    main()
