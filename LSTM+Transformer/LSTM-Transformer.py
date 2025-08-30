import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

import torch

print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from typing import List
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    支持的频率：Y(年)/M(月)/W(周)/D(日)/B(工作日)/H(时)/T(分)/S(秒)
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, timeenc=1, freq="h"):
    if timeenc == 0:
        dates["month"] = dates.date.apply(lambda x: x.month)
        dates["day"] = dates.date.apply(lambda x: x.day)
        dates["weekday"] = dates.date.apply(lambda x: x.weekday())
        dates["hour"] = dates.date.apply(lambda x: x.hour)
        dates["minute"] = dates.date.apply(lambda x: x.minute // 15)
        freq_map = {
            "y": [],
            "m": ["month"],
            "w": ["month"],
            "d": ["month", "day", "weekday"],
            "b": ["month", "day", "weekday"],
            "h": ["month", "day", "weekday", "hour"],
            "t": ["month", "day", "weekday", "hour", "minute"],
        }
        return dates[freq_map[freq.lower()]].values
    elif timeenc == 1:
        dates = pd.to_datetime(dates.date.values)
        features = [feat(dates) for feat in time_features_from_frequency_str(freq)]
        return np.vstack(features).transpose(1, 0)


from layers.Transformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    ConvLayer,
)
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding

try:
    from torch import irfft, rfft
except ImportError:

    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))
        return torch.stack((t.real, t.imag), -1)

    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:, :, 0], x[:, :, 1]), dim=(-d))
        return t.real


def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = rfft(v, 1)
    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    return 2 * V.view(*x_shape)


class dct_channel_block(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2, channel, bias=False),
            nn.Sigmoid(),
        )
        self.dct_norm = nn.LayerNorm([96], eps=1e-6)

    def forward(self, x):
        b, c, l = x.size()
        freq_list = [dct(x[:, i, :]) for i in range(c)]
        stack_dct = torch.stack(freq_list, dim=1)
        lr_weight = self.dct_norm(stack_dct)
        lr_weight = self.fc(lr_weight)
        lr_weight = self.dct_norm(lr_weight)
        return x * lr_weight


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.dropout1(self.conv1(x)))
        out = self.relu(self.dropout2(self.conv2(out)))
        return out


class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dct = dct_channel_block(channel=10)
        self.tcn_layers = nn.ModuleList()
        num_channels = [input_size] + [hidden_size] * (num_layers - 1) + [hidden_size]

        for i in range(num_layers):
            dilation = 2**i
            in_ch, out_ch = num_channels[i], num_channels[i + 1]
            padding = (3 - 1) * dilation // 2
            self.tcn_layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    3,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=0.2,
                )
            )

    def forward(self, x_enc):
        x_enc = dct(x_enc)
        x_enc = x_enc.permute(0, 2, 1)
        for layer in self.tcn_layers:
            x_enc = layer(x_enc)
        return x_enc.permute(0, 2, 1)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x_enc):
        h0 = torch.randn(self.num_layers, x_enc.shape[0], self.hidden_size).to(
            self.device
        )
        c0 = torch.randn(self.num_layers, x_enc.shape[0], self.hidden_size).to(
            self.device
        )
        output, _ = self.lstm(x_enc, (h0, c0))
        return output


class TransformerModel(nn.Module):
    def __init__(self, configs, device):
        super().__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.device = device

        self.lstm = LSTM(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=3,
            batch_size=configs.batch_size,
            device=device,
        )

        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_embed = self.enc_embedding(x_enc, x_mark_enc)
        lstm_feat = self.lstm(x_enc)

        if lstm_feat.shape[-1] != enc_embed.shape[-1]:
            lstm_feat = nn.Linear(lstm_feat.shape[-1], enc_embed.shape[-1])(lstm_feat)

        combined_feat = enc_embed + lstm_feat
        enc_out, _ = self.encoder(combined_feat, attn_mask=None)

        dec_embed = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_embed, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]


plt.rc("font", family="Arial")
plt.style.use("ggplot")


def tslib_data_loader(window, length_size, batch_size, data, data_mark):
    sequence_length = window + length_size
    result = np.array(
        [data[i : i + sequence_length] for i in range(len(data) - sequence_length + 1)]
    )
    result_mark = np.array(
        [
            data_mark[i : i + sequence_length]
            for i in range(len(data) - sequence_length + 1)
        ]
    )

    x_temp = result[:, :-length_size]
    y_temp = result[:, -(length_size + int(window / 2)) :]
    x_temp_mark = result_mark[:, :-length_size]
    y_temp_mark = result_mark[:, -(length_size + int(window / 2)) :]

    x_temp = torch.tensor(x_temp, dtype=torch.float32)
    y_temp = torch.tensor(y_temp, dtype=torch.float32)
    x_temp_mark = torch.tensor(x_temp_mark, dtype=torch.float32)
    y_temp_mark = torch.tensor(y_temp_mark, dtype=torch.float32)

    dataset = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, x_temp, y_temp, x_temp_mark, y_temp_mark


def model_train(
    net,
    train_loader,
    length_size,
    target_dim,
    optimizer,
    criterion,
    num_epochs,
    device,
    print_train=False,
):
    train_loss = []
    print_frequency = max(1, num_epochs // 20)

    for epoch in range(num_epochs):
        net.train()
        total_loss = 0.0

        for batch in train_loader:
            datapoints, labels, dp_mark, lbl_mark = [x.to(device) for x in batch]
            optimizer.zero_grad()

            preds = net(datapoints, dp_mark, labels, lbl_mark)
            labels = labels[:, -length_size:, -target_dim:]

            if preds.shape != labels.shape:
                preds = preds[:, : labels.shape[1], : labels.shape[2]]

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_loss.append(avg_loss)

        if print_train and (epoch + 1) % print_frequency == 0:
            print(f"Epoch: {epoch+1:3d} | Train Loss: {avg_loss:.4f}")

    return net, train_loss, epoch + 1


def model_train_val(
    net,
    train_loader,
    val_loader,
    length_size,
    target_dim,
    optimizer,
    criterion,
    scheduler,
    num_epochs,
    device,
    early_patience=0.15,
    print_train=False,
):
    train_loss = []
    val_loss = []
    print_frequency = max(1, num_epochs // 20)
    early_patience_epochs = int(early_patience * num_epochs)
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(num_epochs):
        net.train()
        total_train_loss = 0.0
        for batch in train_loader:
            datapoints, labels, dp_mark, lbl_mark = [x.to(device) for x in batch]
            optimizer.zero_grad()

            preds = net(datapoints, dp_mark, labels, lbl_mark)
            labels = labels[:, -length_size:, -target_dim:]

            if preds.shape != labels.shape:
                preds = preds[:, : labels.shape[1], : labels.shape[2]]

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss.append(avg_train_loss)

        net.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_x, val_y, val_xm, val_ym = [x.to(device) for x in batch]
                pred_val = net(val_x, val_xm, val_y, val_ym)
                val_y = val_y[:, -length_size:, -target_dim:]

                if pred_val.shape != val_y.shape:
                    pred_val = pred_val[:, : val_y.shape[1], : val_y.shape[2]]

                val_batch_loss = criterion(pred_val, val_y)
                total_val_loss += val_batch_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_loss.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if print_train and (epoch + 1) % print_frequency == 0:
            print(
                f"Epoch: {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            # early_stop_counter += 1      10 epochs
            if early_stop_counter >= early_patience_epochs:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    net.train()
    return net, train_loss, val_loss, epoch + 1


def cal_eval(y_real, y_pred):
    y_real = np.array(y_real).ravel()
    y_pred = np.array(y_pred).ravel()

    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred, squared=True)
    rmse = mean_squared_error(y_real, y_pred, squared=False)
    mae = mean_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred) * 100

    return pd.DataFrame(
        {"R2": [r2], "MSE": [mse], "RMSE": [rmse], "MAE": [mae], "MAPE": [mape]},
        index=["Eval"],
    )


df = pd.read_csv(r"", encoding="gbk")  # 请填写数据文件路径
print("数据形状:", df.shape)
print("数据列名:", df.columns.tolist())

target_dim = 1
window = 10  # 输入序列长度（历史时间步），设为偶数避免分割问题
length_size = 1
batch_size = 64
num_epochs = 50
learning_rate = 0.0002  # 初始学习率0.0001-0.0005
scheduler_patience = int(0.25 * num_epochs)  # Early stopping:10epochs
early_patience = 0.2


data_target = df["Target"]
data_features = df[df.columns.drop("date")]
data_dim = data_features.shape[1]

df_stamp = df[["date"]].copy()
df_stamp["date"] = pd.to_datetime(df_stamp["date"])
data_stamp = time_features(df_stamp, timeenc=1, freq="T")
print("时间特征形状:", data_stamp.shape)

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_features.values)
data_length = len(data_normalized)


train_ratio = 0.8  # 训练集占比
train_size = int(data_length * train_ratio)

data_train = data_normalized[:train_size, :]
data_train_mark = data_stamp[:train_size, :]
data_test = data_normalized[train_size:, :]
data_test_mark = data_stamp[train_size:, :]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class Config:
    def __init__(self):
        self.seq_len = window
        self.label_len = int(window / 2)
        self.pred_len = length_size
        self.freq = "t"

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate  # 学习率:0.0001-0.0005
        self.stop_ratio = early_patience  # 早停比例:0.2

        self.enc_in = data_dim
        self.dec_in = data_dim
        self.c_out = target_dim

        self.d_model = 40
        self.n_heads = 8  # 多头注意力头数8
        self.dropout = 0.1  # dropout比例:0.1
        self.e_layers = 2  # 编码器层数
        self.d_layers = 1  # 解码器层数
        self.d_ff = 40
        self.factor = 5
        self.activation = "gelu"  # 激活函数为'Gelu

        self.embed = "timeF"
        self.output_attention = 0
        self.task_name = "short_term_forecast"


config = Config()

model = TransformerModel(config, device).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=scheduler_patience, verbose=True
)

train_loader, x_train, y_train, x_train_mark, y_train_mark = tslib_data_loader(
    window=window,
    length_size=length_size,
    batch_size=batch_size,
    data=data_train,
    data_mark=data_train_mark,
)

test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(
    window=window,
    length_size=length_size,
    batch_size=batch_size,
    data=data_test,
    data_mark=data_test_mark,
)


trained_model, train_loss, final_epoch = model_train(
    net=model,
    train_loader=train_loader,
    length_size=length_size,
    target_dim=target_dim,
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=num_epochs,
    device=device,
    print_train=True,
)

trained_model.eval()
with torch.no_grad():
    pred = trained_model(
        x_test.to(device),
        x_test_mark.to(device),
        y_test.to(device),
        y_test_mark.to(device),
    )
    true = y_test[:, -length_size:, -target_dim:]

true = true.detach().cpu()
pred = pred.detach().cpu()
print("\n预测前维度 - True:", true.shape, "Pred:", pred.shape)

if pred.shape != true.shape:
    pred = pred[:, : true.shape[1], : true.shape[2]]

print("预测后维度 - True:", true.shape, "Pred:", pred.shape)

target_scaler = MinMaxScaler()
target_scaler.fit(data_target.values.reshape(-1, 1))

true_unscale = target_scaler.inverse_transform(true[:, -1, :].numpy().reshape(-1, 1))
pred_unscale = target_scaler.inverse_transform(pred[:, -1, :].numpy().reshape(-1, 1))


eval_df = cal_eval(true_unscale, pred_unscale)
print("\n评估指标:")
print(eval_df.round(4))

df_pred_true = pd.DataFrame(
    {"Predict": pred_unscale.flatten(), "Real": true_unscale.flatten()}
)
plt.figure(figsize=(12, 4))
plt.plot(df_pred_true["Real"], label="Real Value", linewidth=1.5)
plt.plot(df_pred_true["Predict"], label="Predicted Value", linewidth=1.5, alpha=0.8)
plt.title("LSTM-Transformer Prediction Result (总磷)", fontsize=12)
plt.xlabel("Time Step", fontsize=10)
plt.ylabel("Value", fontsize=10)
plt.legend()
plt.tight_layout()
plt.show()


result_df = pd.DataFrame(
    {"真实值": true_unscale.flatten(), "预测值": pred_unscale.flatten()}
)
result_df.to_csv("真实值与预测值 wubugou LT-总磷.csv", index=False, encoding="utf-8")
print("\n真实值和预测值已保存到CSV文件")
