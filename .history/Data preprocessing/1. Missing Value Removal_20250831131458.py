
import pandas as pd


file_path = ''  
df = pd.read_excel(file_path)

df_cleaned = df.dropna()


output_file_path = ''  
df_cleaned.to_csv(output_file_path, index=False)

print(f"清理后的数据已保存至: {output_file_path}")
