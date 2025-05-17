# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
#
# input_file = "/home/israt/OMNETPP/ts/simu5G/src/data/tower_Load_test.txt"
# output_file_4g = "/home/israt/OMNETPP/ts/simu5G/src/data/outputLSTM_4G.txt"
# output_file_5g = "/home/israt/OMNETPP/ts/simu5G/src/data/outputLSTM_5G.txt"
# n_steps = 10
#
# try:
#     df = pd.read_csv(input_file, header=None, names=['Time', 'TowerId', 'Type', 'AvgLoad'])
# except FileNotFoundError:
#     print(f"Error: {input_file} not found")
#     exit(1)
#
# df_4g = df[df['Type'] == 0]['AvgLoad'].values
# df_5g = df[df['Type'] == 1]['AvgLoad'].values
#
# scaler_4g = MinMaxScaler(feature_range=(0, 1))
# scaler_5g = MinMaxScaler(feature_range=(0, 1))
# df_4g_scaled = scaler_4g.fit_transform(df_4g.reshape(-1, 1)).flatten()
# df_5g_scaled = scaler_5g.fit_transform(df_5g.reshape(-1, 1)).flatten()
#
# if len(df_4g) < n_steps + 1 or len(df_5g) < n_steps + 1:
#     print(f"Insufficient data - 4G: {len(df_4g)}, 5G: {len(df_5g)}")
#     prediction_4g = prediction_5g = 0.1
# else:
#     def split_sequence(sequence, n_steps):
#         X, y = [], []
#         for i in range(len(sequence) - n_steps):
#             X.append(sequence[i:i + n_steps])
#             y.append(sequence[i + n_steps])
#         return np.array(X), np.array(y)
#
#     X_4g, y_4g = split_sequence(df_4g_scaled, n_steps)
#     X_5g, y_5g = split_sequence(df_5g_scaled, n_steps)
#     X_4g = X_4g.reshape((X_4g.shape[0], X_4g.shape[1], 1))
#     X_5g = X_5g.reshape((X_5g.shape[0], X_5g.shape[1], 1))
#
#     def train_lstm(X, y):
#         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
#         model = Sequential([
#             LSTM(10, activation='tanh', input_shape=(n_steps, 1), return_sequences=True),
#             Dropout(0.1),
#             LSTM(10, activation='tanh'),
#             Dense(5, activation='tanh'),
#             Dense(1, activation='sigmoid')  # Cap output to [0, 1]
#         ])
#         model.compile(optimizer='adam', loss='mse')
#         model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32, verbose=0)
#         return model
#
#     model_4g = train_lstm(X_4g, y_4g)
#     model_5g = train_lstm(X_5g, y_5g)
#
#     test_4g = df_4g_scaled[-n_steps:].reshape((1, n_steps, 1))
#     test_5g = df_5g_scaled[-n_steps:].reshape((1, n_steps, 1))
#     prediction_4g_scaled = model_4g.predict(test_4g, verbose=0)[0][0]
#     prediction_5g_scaled = model_5g.predict(test_5g, verbose=0)[0][0]
#
#     prediction_4g = scaler_4g.inverse_transform([[prediction_4g_scaled]])[0][0]
#     prediction_5g = scaler_5g.inverse_transform([[prediction_5g_scaled]])[0][0]
#
#     prediction_4g = min(max(prediction_4g, 0.0), 1.0)
#     prediction_5g = min(max(prediction_5g, 0.0), 1.0)
#
# with open(output_file_4g, 'w') as f:
#     f.write(f"{prediction_4g:.6f}")
# with open(output_file_5g, 'w') as f:
#     f.write(f"{prediction_5g:.6f}")
#
# print(f"Predicted 4G Load: {prediction_4g:.6f}")
# print(f"Predicted 5G Load: {prediction_5g:.6f}")
#




import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input_file = "/home/israt/OMNETPP/ts/simu5G/src/data/tower_Load_test.txt"
output_file_4g = "/home/israt/OMNETPP/ts/simu5G/src/data/outputLSTM_4G.txt"
output_file_5g = "/home/israt/OMNETPP/ts/simu5G/src/data/outputLSTM_5G.txt"
n_steps = 5

def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence) - n_steps):
        X.append(sequence[i:i + n_steps])
        y.append(sequence[i + n_steps])
    return np.array(X), np.array(y)

print(f"Checking if input file exists: {input_file}")
if not os.path.exists(input_file):
    print(f"Error: Input file {input_file} not found")
    with open(output_file_4g, 'w') as f:
        f.write("0.1")
    with open(output_file_5g, 'w') as f:
        f.write("0.1")
    exit(1)

print(f"Reading input file: {input_file}")
try:
    with open(input_file, 'r') as f:
        print(f"File contents:\n{f.read()}")
    df = pd.read_csv(input_file, header=None, names=['Time', 'TowerId', 'Type', 'AvgLoad'])
    print(f"Input data shape: {df.shape}")
    print(f"Unique TowerIds: {df['TowerId'].unique()}")
    print(f"Unique Types: {df['Type'].unique()}")
except Exception as e:
    print(f"Error reading CSV: {e}")
    with open(output_file_4g, 'w') as f:
        f.write("0.1")
    with open(output_file_5g, 'w') as f:
        f.write("0.1")
    exit(1)

predictions_4g = {}
predictions_5g = {}

for tower_id in range(1, 7):  # Process towers 1–6
    for type_val in [0, 1]:
        tower_data = df[(df['TowerId'] == tower_id) & (df['Type'] == type_val)]['AvgLoad'].values
        print(f"Tower {tower_id}, Type {type_val}: {len(tower_data)} data points")
        if len(tower_data) < n_steps + 1:
            print(f"Insufficient data for Tower {tower_id}, Type {type_val}")
            pred = np.mean(tower_data) if len(tower_data) > 0 else 0.1
            if type_val == 0:
                predictions_4g[tower_id] = pred
            else:
                predictions_5g[tower_id] = pred
            print(f"Fallback prediction for Tower {tower_id}, Type {type_val}: {pred:.6f}")
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(tower_data.reshape(-1, 1)).flatten()

        X, y = split_sequence(scaled_data, n_steps)
        print(f"Tower {tower_id}, Type {type_val}: X shape {X.shape}, y shape {y.shape}")
        if X.shape[0] == 0:
            print(f"No sequences for Tower {tower_id}, Type {type_val}")
            pred = np.mean(tower_data) if len(tower_data) > 0 else 0.1
            if type_val == 0:
                predictions_4g[tower_id] = pred
            else:
                predictions_5g[tower_id] = pred
            print(f"Fallback prediction for Tower {tower_id}, Type {type_val}: {pred:.6f}")
            continue
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential([
            LSTM(3, activation='tanh', input_shape=(n_steps, 1)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='mse')
        print(f"Training model for Tower {tower_id}, Type {type_val}")
        try:
            model.fit(X, y, epochs=1, batch_size=16, verbose=0)
        except Exception as e:
            print(f"Error training model for Tower {tower_id}, Type {type_val}: {e}")
            pred = np.mean(tower_data) if len(tower_data) > 0 else 0.1
            if type_val == 0:
                predictions_4g[tower_id] = pred
            else:
                predictions_5g[tower_id] = pred
            print(f"Fallback prediction for Tower {tower_id}, Type {type_val}: {pred:.6f}")
            continue

        test_data = scaled_data[-n_steps:].reshape((1, n_steps, 1))
        try:
            pred_scaled = model.predict(test_data, verbose=0)[0][0]
            pred = scaler.inverse_transform([[pred_scaled]])[0][0]
            pred = min(max(pred, 0.0), 1.0)
        except Exception as e:
            print(f"Error predicting for Tower {tower_id}, Type {type_val}: {e}")
            pred = np.mean(tower_data) if len(tower_data) > 0 else 0.1
            if type_val == 0:
                predictions_4g[tower_id] = pred
            else:
                predictions_5g[tower_id] = pred
            print(f"Fallback prediction for Tower {tower_id}, Type {type_val}: {pred:.6f}")
            continue

        if type_val == 0:
            predictions_4g[tower_id] = pred
        else:
            predictions_5g[tower_id] = pred
        print(f"Predicted Load for Tower {tower_id}, Type {'4G' if type_val == 0 else '5G'}: {pred:.6f}")

avg_4g = np.mean(list(predictions_4g.values())) if predictions_4g else 0.1
avg_5g = np.mean(list(predictions_5g.values())) if predictions_5g else 0.1
print(f"Average 4G prediction: {avg_4g:.6f}, Average 5G prediction: {avg_5g:.6f}")
with open(output_file_4g, 'w') as f:
    f.write(f"{avg_4g:.6f}")
with open(output_file_5g, 'w') as f:
    f.write(f"{avg_5g:.6f}")
