import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import torch
print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

plt.rc('font',family='Arial')
plt.style.use("ggplot")


from models import iTransformer
from utils.timefeatures import time_features

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def tslib_data_loader(window, length_size, batch_size, data, data_mark):
    

    
    seq_len = window
    sequence_length = seq_len + length_size
    result = np.array([data[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])
    result_mark = np.array([data_mark[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])

    
    x_temp = result[:, :-length_size]
    y_temp = result[:, -(length_size + int(window / 2)):]

    x_temp_mark = result_mark[:, :-length_size]
    y_temp_mark = result_mark[:, -(length_size + int(window / 2)):]

    
    x_temp = torch.tensor(x_temp).type(torch.float32)
    x_temp_mark = torch.tensor(x_temp_mark).type(torch.float32)
    y_temp = torch.tensor(y_temp).type(torch.float32)
    y_temp_mark = torch.tensor(y_temp_mark).type(torch.float32)

    ds = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dataloader, x_temp, y_temp, x_temp_mark, y_temp_mark


def model_train(net, train_loader, length_size, optimizer, criterion, num_epochs, device, print_train=False):
    

    train_loss = []  
    print_frequency = num_epochs / 20  

    for epoch in range(num_epochs):
        total_train_loss = 0  

        net.train()  
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(train_loader):
            datapoints, labels, datapoints_mark, labels_mark = datapoints.to(device), labels.to(
                device), datapoints_mark.to(device), labels_mark.to(device)
            optimizer.zero_grad()  
            preds = net(datapoints, datapoints_mark, labels, labels_mark, None).squeeze()  
            labels = labels[:, -length_size:].squeeze()
            loss = criterion(preds, labels)  
            loss.backward()  
            optimizer.step()  
            total_train_loss += loss.item()  

        avg_train_loss = total_train_loss / len(train_loader)  
        train_loss.append(avg_train_loss)  

        
        if print_train:
            if (epoch + 1) % print_frequency == 0:
                print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

    return net, train_loss, epoch + 1
def model_train_val(net, train_loader, val_loader, length_size, optimizer, criterion, scheduler, num_epochs, device, early_patience=0.15, print_train=False):
    

    train_loss = []  
    val_loss = []  
    print_frequency = num_epochs / 20  

    early_patience_epochs = int(early_patience * num_epochs)  #Early stopping 10
    best_val_loss = float('inf')  
    early_stop_counter = 0  

    for epoch in range(num_epochs):
        total_train_loss = 0  

        net.train()  
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(train_loader):
            datapoints, labels, datapoints_mark, labels_mark = datapoints.to(device), labels.to(device), datapoints_mark.to(device), labels_mark.to(device)
            optimizer.zero_grad()  
            preds = net(datapoints, datapoints_mark, labels, labels_mark, None).squeeze()  
            labels = labels[:, -length_size:].squeeze()  
            loss = criterion(preds, labels)  
            loss.backward()  
            optimizer.step()  
            total_train_loss += loss.item()  

        avg_train_loss = total_train_loss / len(train_loader)  
        train_loss.append(avg_train_loss)  

        with torch.no_grad():  
            total_val_loss = 0
            for val_x, val_y, val_x_mark, val_y_mark in val_loader:
                val_x, val_y, val_x_mark, val_y_mark = val_x.to(device), val_y.to(device), val_x_mark.to(device), val_y_mark.to(device)  
                pred_val_y = net(val_x, val_x_mark, val_y, val_y_mark, None).squeeze()  
                val_y = val_y[:, -length_size:].squeeze()  
                val_loss_batch = criterion(pred_val_y, val_y)  
                total_val_loss += val_loss_batch.item()

            avg_val_loss = total_val_loss / len(val_loader)  
            val_loss.append(avg_val_loss)  

            scheduler.step(avg_val_loss)  

        
        if print_train == True:
            if (epoch + 1) % print_frequency == 0:
                print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0  
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_patience_epochs:
                print(f'Early stopping triggered at epoch {epoch + 1}.')
                break  

    net.train()  
    return net, train_loss, val_loss, epoch + 1



def cal_eval(y_real, y_pred):
    

    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()

    r2 = r2_score(y_real, y_pred)
     
    mse = mean_squared_error(y_real, y_pred)

    
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred) * 100  

    df_eval = pd.DataFrame({'R2': r2,
                            'MSE': mse, 'RMSE': rmse,
                            'MAE': mae, 'MAPE': mape},
                           index=['Eval'])

    return df_eval

df = pd.read_csv(r'',encoding='gbk') #数据集路径

data_dim = df[df.columns.drop('date')].shape[1]  
data_target = df['Target']  
data = df[df.columns.drop('date')]  

df_stamp = df[['date']]
df_stamp.loc[:, 'date'] = pd.to_datetime(df_stamp.date)
data_stamp = time_features(df_stamp, timeenc=1, freq='T')  





scaler = MinMaxScaler()
data_inverse = scaler.fit_transform(np.array(data))

data_length = len(data_inverse)
train_set = 0.8

data_train = data_inverse[:int(train_set * data_length), :]  
data_train_mark = data_stamp[:int(train_set * data_length), :]
data_test = data_inverse[int(train_set * data_length):, :]  
data_test_mark = data_stamp[int(train_set * data_length):, :]

n_feature = data_dim

window = 10  #Time window 10
length_size = 1  
batch_size = 80
train_loader, x_train, y_train, x_train_mark, y_train_mark = tslib_data_loader(window, length_size, batch_size, data_train, data_train_mark)
test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(window, length_size, batch_size, data_test, data_test_mark)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs =100  
learning_rate = 0.0004  #Learning rate在0.0001-0.0005之间
scheduler_patience = int(0.25 * num_epochs)  
early_patience = 0.16 


class Config:
    def __init__(self):
        
        self.seq_len = window  
        self.label_len = int(window / 2)  
        self.pred_len = length_size  
        self.freq = 't'  
        
        self.batch_size = batch_size  
        self.num_epochs = num_epochs  
        self.learning_rate = learning_rate  
        self.stop_ratio = early_patience  
        
        self.dec_in = data_dim  
        self.enc_in = data_dim  
        self.c_out = 1  
        
        self.d_model =99  
        self.n_heads = 8  #Head numbers 8
        self.dropout = 0.1  #Dropout 0.1
        self.e_layers = 2  
        self.d_layers = 1  
        self.d_ff =102 
        self.factor = 5  
        self.activation = 'gelu'  #Activation gelu
        self.channel_independence = 0  

        self.top_k = 5  
        self.num_kernels = 6  
        self.distil = 1  
        
        self.embed = 'timeF'  
        self.output_attention = 0  
        self.task_name = 'short_term_forecast'  
        self.moving_avg = window - 1  


config = Config()

model_type = 'LSTM+iTransformer'
net = iTransformer.Model(config).to(device)
criterion = nn.MSELoss().to(device)  
optimizer = optim.Adam(net.parameters(), lr=learning_rate)  
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_patience)


trained_model, train_loss, final_epoch = model_train(net, train_loader, length_size, optimizer, criterion, num_epochs, device, print_train=True)


trained_model.eval()  

pred = trained_model(x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device))
true = y_test[:,-length_size:,-1:].detach().cpu()
pred = pred.detach().cpu()

print("Shape of true before adjustment:", true.shape)
print("Shape of pred before adjustment:", pred.shape)


true = true[:, :, -1]
pred = pred[:, :, -1]  



print("Shape of pred after adjustment:", pred.shape)
print("Shape of true after adjustment:", true.shape)

 
y_data_test_inverse = scaler.fit_transform(np.array(data_target).reshape(-1, 1))
pred_uninverse = scaler.inverse_transform(pred[:, -1:])    
true_uninverse = scaler.inverse_transform(true[:, -1:])

true, pred = true_uninverse, pred_uninverse

df_eval = cal_eval(true, pred)  
print(df_eval)

df_pred_true = pd.DataFrame({'Predict': pred.flatten(), 'Real': true.flatten()})
df_pred_true.plot(figsize=(12, 4))
plt.title(model_type + ' Result')
plt.savefig('总磷.png')
plt.show()



result_df = pd.DataFrame({'真实值': true.flatten(), '预测值': pred.flatten()})

result_df.to_csv('真实值与预测值.csv', index=False, encoding='utf-8')

print('真实值和预测值已保存到真实值与预测值.csv文件中。')