import pandas as pd
from src.data_pipeline.adaptive_engine import detect_dataset_type, perform_eda_and_clean_dataset

datasets = [
    'data/sample_batting_summary_2026.csv',
    'data/sample_bowling_summary_2026.csv',
    'data/sample_all_rounder_summary_2026.csv',
    'data/sample_points_table_2026.csv',
    'data/sample_custom_tabular_dataset.csv'
]

for d in datasets:
    df = pd.read_csv(d)
    dtype = detect_dataset_type(df)
    print(f"=== Testing {d} ===")
    print(f"Detected type: {dtype}")
    cleaned, meta = perform_eda_and_clean_dataset(df, dtype)
    print(f"Cleaned rows: {len(cleaned)}, Cols: {len(cleaned.columns)}")
    print(f"ML Model: {meta['ml_report'].get('model_type', 'None')}")
    print("OK!\n")
