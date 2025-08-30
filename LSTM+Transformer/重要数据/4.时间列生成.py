import pandas as pd

start_time = "2016-07-01 00:00"
end_time = "2016-8-26 00:00"
time_range = pd.date_range(start=start_time, end=end_time, freq="15T")

df = pd.DataFrame(time_range, columns=["date"])

df.to_csv("time_series.csv", index=False)
