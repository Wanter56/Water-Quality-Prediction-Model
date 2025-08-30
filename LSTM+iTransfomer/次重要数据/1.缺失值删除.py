
import pandas as pd

# 读取CSV文件
file_path = ''  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 删除存在缺失值的行
df_cleaned = df.dropna()

# 输出到一个新文件
output_file_path = ''  # 替换为你的输出文件路径
df_cleaned.to_csv(output_file_path, index=False)

# 输出结果
print(f"清理后的数据已保存至: {output_file_path}")
