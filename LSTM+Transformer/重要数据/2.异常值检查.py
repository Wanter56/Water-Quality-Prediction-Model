import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("团结闸-总磷-clean.csv", encoding="gbk")

last_column = df.iloc[:, -1]

mean_value = last_column.mean()

upper_threshold = mean_value * 2
lower_threshold = mean_value * 0.5

print(upper_threshold)
print(lower_threshold)

df_cleaned = df[(last_column >= lower_threshold) & (last_column <= upper_threshold)]

df_cleaned.to_csv("团结闸-总磷-clean.csv", index=False)

plt.plot(df_cleaned.iloc[:, -1], label="Cleaned Data")
plt.title("Cleaned Data Visualization")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.show()
