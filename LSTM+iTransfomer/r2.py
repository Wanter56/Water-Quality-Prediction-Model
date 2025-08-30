import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


file_path = r''  
df = pd.read_csv(file_path)



column1 = '真实值'
column2 = '预测值'


if column1 in df.columns and column2 in df.columns:
    
    data1 = df[column1]
    data2 = df[column2]

    
    r2 = r2_score(data1, data2)

    
    mae = mean_absolute_error(data1, data2)

    
    mse = mean_squared_error(data1, data2)

    plt.rcParams['figure.dpi'] = 300

    
    plt.figure(figsize=(12, 6))
    plt.plot(data1, label='真实值', color='blue')
    plt.plot(data2, label='预测值', color='red')
    plt.ylim(0.05, 0.15)
    plt.legend()

    
    plt.grid(True)

    
    plt.show()

    
    print(f"R² (决定系数): {r2}")
    print(f"MAE (平均绝对误差): {mae}")
    print(f"MSE (均方误差): {mse}")
else:
    print(f"指定的列 {column1} 或 {column2} 不存在于文件中。")