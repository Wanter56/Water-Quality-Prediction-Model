import pandas as pd

file_path = "三兴村-总氮.xlsx"
df = pd.read_excel(file_path)

df_cleaned = df.dropna()

output_file_path = "三兴村-总氮-clean.csv"
df_cleaned.to_csv(output_file_path, index=False)

print(f"清理后的数据已保存至: {output_file_path}")
