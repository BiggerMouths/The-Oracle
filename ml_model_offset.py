import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('中美汇率.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=False, errors='coerce')
df['day_of_week'] = df['Date'].dt.dayofweek
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7).round(3)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7).round(3)

feature_cols = ['rate', 'dow_sin', 'dow_cos']
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Save the scaler for later use
joblib.dump(scaler, 'scaler.joblib')

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, offset, feature_cols=None):
        self.seq_length = seq_length
        self.offset = offset
        self.feature_cols = feature_cols
        self.X = []
        self.y = []
        for i in range(len(data) - seq_length - offset + 1):
            X_seq = data.loc[i:i+seq_length-1, feature_cols].values
            y_value = data.loc[i+seq_length - 1 + offset, 'rate']
            self.X.append(X_seq)
            self.y.append(y_value)
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GRUForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUForecast, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seq_length = 14
offset = 7

# Enable safe loading of GRU, Linear and GRUForecast model
from torch.nn.modules.linear import Linear
torch.serialization.add_safe_globals([torch.nn.modules.rnn.GRU, Linear, GRUForecast])

model = torch.load('RMB_USD.pth', map_location=torch.device('cpu'), weights_only=False)
model.eval()

N = seq_length + offset + 7 - 1
pred_input = df.iloc[-N:].reset_index(drop=True)


pred_dataset = TimeSeriesDataset(pred_input, seq_length=seq_length, offset=offset, feature_cols=feature_cols)
pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False)

predictions = []
model.eval()
with torch.no_grad():
    for X_batch, _ in pred_loader:
        X_batch = X_batch.to(model.fc.weight.device)
        y_pred = model(X_batch)
        predictions.append(y_pred.cpu().numpy()[0][0])
predictions = np.array(predictions)

all_zeros = np.zeros((len(predictions), len(feature_cols)))
all_zeros[:, 0] = predictions
inversed_predictions = scaler.inverse_transform(all_zeros)[:, 0]

df2 = pd.read_csv('中欧汇率.csv')
df2['Date'] = pd.to_datetime(df2['Date'], dayfirst=False, errors='coerce')
df2['day_of_week'] = df2['Date'].dt.dayofweek
df2['dow_sin'] = np.sin(2 * np.pi * df2['day_of_week'] / 7).round(3)
df2['dow_cos'] = np.cos(2 * np.pi * df2['day_of_week'] / 7).round(3)

df2[feature_cols] = scaler.transform(df2[feature_cols])

model = torch.load('RMB_EURO.pth', map_location=torch.device('cpu'))
model.eval()

N2 = seq_length + offset + 7 - 1
pred_input2 = df2.iloc[-N2:].reset_index(drop=True)
pred_dataset2 = TimeSeriesDataset(pred_input2, seq_length=seq_length, offset=offset, feature_cols=feature_cols)
pred_loader2 = DataLoader(pred_dataset2, batch_size=1, shuffle=False)

predictions2 = []
model.eval()
with torch.no_grad():
    for X_batch, _ in pred_loader2:
        X_batch = X_batch.to(device)
        y_pred = model(X_batch)
        predictions2.append(y_pred.cpu().numpy()[0][0])
predictions2 = np.array(predictions2)

all_zeros = np.zeros((len(predictions2), len(feature_cols)))
all_zeros[:, 0] = predictions2
inversed_predictions2 = scaler.inverse_transform(all_zeros)[:, 0]

last_date = df['Date'].max()

future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)

future_df = pd.DataFrame({'Date': future_dates, 'USD': inversed_predictions, 'EURO': inversed_predictions2})

future_df = future_df.set_index('Date')

print(future_df)