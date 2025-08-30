import pandas as pd

# 生成时间范围
start_time = '2016-07-01 00:00'
end_time = '2016-10-26 00:00'
time_range = pd.date_range(start=start_time, end=end_time, freq='15T')

# 将时间范围转换为 DataFrame
df = pd.DataFrame(time_range, columns=['Time'])

# 保存为 CSV 文件
df.to_csv('time_series.csv', index=False)