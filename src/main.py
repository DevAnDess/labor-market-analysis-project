from src.data_collection import fetch_all_vacancies
from src.data_processing import process_data
from src.data_analysis import analyze_data

def main():
    raw_data = fetch_all_vacancies()
    clean_data = process_data(raw_data)
    analysis_results = analyze_data(clean_data)
    print(analysis_results.head())
    print(clean_data.head())


if __name__ == "__main__":
    main()
