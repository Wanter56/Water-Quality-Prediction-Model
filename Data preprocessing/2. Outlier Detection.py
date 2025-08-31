import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('',encoding='gbk')

# 获取最后一列数据
last_column = df.iloc[:, -1]

# 计算该列的均值
mean_value = last_column.mean()

# 计算阈值，超过均值5倍或者小于均值0.5倍的行将被删除
upper_threshold = mean_value * 2
lower_threshold = mean_value * 0.5

print(upper_threshold)
print(lower_threshold)
# 删除异常值行
df_cleaned = df[(last_column >= lower_threshold) & (last_column <= upper_threshold)]

# 保存到新的CSV文件
df_cleaned.to_csv('', index=False)

# 可视化最后一列数据
plt.plot(df_cleaned.iloc[:, -1], label='Cleaned Data')
plt.title('Cleaned Data Visualization')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()
