import pandas as pd
df = pd.read_parquet("hf://datasets/TAAC2026/data_sample_1000/demo_1000.parquet")
print(df.head(1).to_string())