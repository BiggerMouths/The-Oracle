
import socket
import torch
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

# Define model class
class TimeSeriesDataset(torch.utils.data.Dataset):
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

class GRUForecast(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=1):
        super(GRUForecast, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Enable safe loading of GRU, Linear and GRUForecast model
from torch.nn.modules.linear import Linear
torch.serialization.add_safe_globals([torch.nn.modules.rnn.GRU, Linear, GRUForecast])

# Load pretrained models and scaler
usd_model = torch.load('RMB_USD.pth', map_location=torch.device('cpu'))
eur_model = torch.load('RMB_EURO.pth', map_location=torch.device('cpu'))
scaler = joblib.load('scaler.joblib')
usd_model.eval()
eur_model.eval()

def predict_next_week():
    # USD predictions
    df = pd.read_csv('中美汇率.csv')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=False, errors='coerce')
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7).round(3)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7).round(3)

    feature_cols = ['rate', 'dow_sin', 'dow_cos']
    df[feature_cols] = scaler.transform(df[feature_cols])

    seq_length = 14
    offset = 7
    N = seq_length + offset + 7 - 1
    pred_input = df.iloc[-N:].reset_index(drop=True)
    
    pred_dataset = TimeSeriesDataset(pred_input, seq_length=seq_length, offset=offset, feature_cols=feature_cols)
    pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=1, shuffle=False)

    # EUR predictions
    df2 = pd.read_csv('中欧汇率.csv')
    df2['Date'] = pd.to_datetime(df2['Date'], dayfirst=False, errors='coerce')
    df2['day_of_week'] = df2['Date'].dt.dayofweek
    df2['dow_sin'] = np.sin(2 * np.pi * df2['day_of_week'] / 7).round(3)
    df2['dow_cos'] = np.cos(2 * np.pi * df2['day_of_week'] / 7).round(3)
    
    df2[feature_cols] = scaler.transform(df2[feature_cols])
    
    N2 = seq_length + offset + 7 - 1
    pred_input2 = df2.iloc[-N2:].reset_index(drop=True)
    pred_dataset2 = TimeSeriesDataset(pred_input2, seq_length=seq_length, offset=offset, feature_cols=feature_cols)
    pred_loader2 = torch.utils.data.DataLoader(pred_dataset2, batch_size=1, shuffle=False)

    predictions_dict = {}
    future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=7)

    with torch.no_grad():
        for date, ((X_batch, _), (X_batch2, _)) in zip(future_dates, zip(pred_loader, pred_loader2)):
            X_batch = X_batch.to(usd_model.fc.weight.device)
            X_batch2 = X_batch2.to(eur_model.fc.weight.device)
            
            usd_pred = usd_model(X_batch)
            eur_pred = eur_model(X_batch2)
            
            # Convert predictions back using scaler
            all_zeros = np.zeros((1, 3))
            
            all_zeros[0, 0] = usd_pred[0].item()
            usd_rate = scaler.inverse_transform(all_zeros)[0, 0]
            
            all_zeros[0, 0] = eur_pred[0].item()
            eur_rate = scaler.inverse_transform(all_zeros)[0, 0]
            
            predictions_dict[date.strftime('%Y-%m-%d')] = {
                'USD': usd_rate,
                'EUR': eur_rate
            }

    return predictions_dict

def handle_query(query):
    predictions = predict_next_week()

    if query.lower() == "hello":
        return "Hello, I'm the Oracle. How can I help you today?"

    # Check if query contains a date
    try:
        # Try to parse the date from query
        query_date = datetime.strptime(query.strip(), '%Y-%m-%d').strftime('%Y-%m-%d')
        if query_date in predictions:
            rates = predictions[query_date]
            return f"Exchange rates prediction for {query_date}:\nUSD/CNY: {rates['USD']:.4f}\nEUR/CNY: {rates['EUR']:.4f}"
    except ValueError:
        pass

    if "next week" in query.lower() or "下周" in query:
        predictions = predict_next_week()
        response = "Exchange Rate Predictions for Next Week:\n\n"
        response += "Date         | CNY/USD  | CNY/EUR\n"
        response += "-------------|----------|----------\n"

        # Get next 7 days starting from tomorrow
        current_date = datetime.now()
        start_date = current_date + timedelta(days=1)
        week_dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]

        # Show predictions for all 7 days
        for date in week_dates:
            if date in predictions:
                rates = predictions[date]
                response += f"{date} | {rates['USD']:.4f} | {rates['EUR']:.4f}\n"

        return response.strip()

    return "I'm the Oracle. You can ask me about exchange rates for specific dates next week, or request predictions for the entire week."

def start_server(host='0.0.0.0', port=9999):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Oracle is ready, listening on {host}:{port} ...")

    while True:
        client_socket, addr = server_socket.accept()
        print("Connection from:", addr)
        try:
            data = client_socket.recv(1024)
            if not data:
                continue
            request = data.decode()

            if request.startswith('GET') or request.startswith('POST'):
                response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nPlease use the socket client to connect."
                client_socket.send(response.encode())
                continue

            print("Received query:", request)
            response = handle_query(request)
            client_socket.send(response.encode())
        except Exception as e:
            print("Error processing request:", e)
        finally:
            client_socket.close()

if __name__ == "__main__":
    start_server()
