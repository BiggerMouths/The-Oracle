# The Oracle

The Oracle is an experimental Python chatbot for exchange-rate forecasting. It
uses pre-trained gated recurrent unit (GRU) models to estimate RMB values for
the US dollar and euro, then serves the results through a small TCP server and
an interactive command-line client.

The project uses historical data from the repository. It does not download live
exchange rates, and its output should not guide financial decisions.

## Features

- Forecasts seven dates for USD and EUR from the included datasets
- Uses 14 observations as the input sequence with a seven-day forecast offset
- Accepts a date in `YYYY-MM-DD` format or a request for `next week` / `下周`
- Runs on a plain TCP socket at `0.0.0.0:9999` by default
- Includes pre-trained models, source data and a standalone prediction script

## How it works

1. `server.py` loads both GRU models, the scaler and the two CSV datasets.
2. The server derives day-of-week features and prepares 14-day input sequences.
3. Each model produces seven rate estimates with a seven-day offset.
4. `client.py` sends your query to the server and prints the response.

The client opens a new TCP connection for each message. The server recognises
HTTP-shaped requests, but it only returns a text notice asking you to use the
socket client. This project does not provide an HTTP API.

## Requirements

- Python 3
- PyTorch
- NumPy
- pandas
- joblib
- scikit-learn
- Matplotlib, if you want to run `ml_model_offset.py`

The repository does not include pinned dependency versions. Create a virtual
environment and install the packages with:

```bash
git clone https://github.com/BiggerMouths/The-Oracle.git
cd The-Oracle
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install torch numpy pandas joblib scikit-learn matplotlib
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Running the project

Run all commands from the repository root because the Python files load the
models and datasets with relative paths.

Start the server:

```bash
python3 server.py
```

The server should print:

```text
Oracle is ready, listening on 0.0.0.0:9999 ...
```

Open another terminal, activate the same virtual environment and start the
client:

```bash
python3 client.py
```

The client sends `hello` when it connects. You can then enter:

```text
next week
下周
2025-03-24
```

Enter `quit`, `exit` or `bye` to close the client.

## Data and rate units

The bundled datasets end on **23 March 2025**. The server therefore creates
forecast dates from **24 March 2025 to 30 March 2025**, regardless of the date
on which you run it.

The CSV files store rates on a scale of roughly 700 to 900. This README treats
those values as the amount of CNY per 100 units of USD or EUR, which matches
that scale. The files do not include a units column or other schema metadata,
so check the original data source before interpreting the numbers.

The response labels are inconsistent in the current code: single-date results
use `USD/CNY` and `EUR/CNY`, while the weekly table uses `CNY/USD` and
`CNY/EUR`. The implementation does not convert between those directions.

## Repository contents

| File | Purpose |
| --- | --- |
| `server.py` | Loads the models and data, generates forecasts and serves TCP requests |
| `client.py` | Provides the interactive command-line client |
| `ml_model_offset.py` | Runs a standalone prediction workflow and rebuilds the saved scaler from the USD dataset |
| `RMB_USD.pth` | Pre-trained GRU model for the USD dataset |
| `RMB_EURO.pth` | Pre-trained GRU model for the EUR dataset |
| `scaler.joblib` | Saved feature scaler used by the server |
| `中美汇率.csv` | Historical USD/RMB data ending on 23 March 2025 |
| `中欧汇率.csv` | Historical EUR/RMB data ending on 23 March 2025 |

## Known limitations

- The server uses static historical data and cannot forecast beyond the seven
  dates that follow the final data point.
- The `next week` response filters those forecasts using the computer's current
  date. If the current week does not overlap 24–30 March 2025, the response may
  contain a heading with no rate rows.
- The project provides no training metrics, backtesting results or accuracy
  claims.
- The TCP protocol has no authentication or encryption. Binding to `0.0.0.0`
  exposes the service to reachable network interfaces unless a firewall blocks
  the port.
- The training utility fits the saved scaler to the USD dataset and reuses it
  for the EUR data.
- The query parser accepts only the phrases and date format documented above.

## Financial disclaimer

This repository is a learning project. Its forecasts may be stale, inaccurate
or based on an incorrect interpretation of the source data. Check current rates
with a reliable provider and seek qualified advice before making a financial
decision.
