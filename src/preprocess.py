import pandas as pd

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df["test_result"] = df["test_result"].map({"negative": 0, "positive": 1})
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

