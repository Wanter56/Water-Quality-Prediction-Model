import pandas as pd

file_path = "三兴村-总氮.xlsx"
df = pd.read_excel(file_path)

missing_values = df.isnull().sum()

nan_values = df.isna().sum()

print("缺失值检查：")
print(missing_values)
print("\nNaN值检查：")
print(nan_values)

missing_values_location = df[df.isnull().any(axis=1)]

print("缺失值和NaN值的位置：")
print(missing_values_location)
