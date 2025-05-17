# import numpy as np
# import pandas as pd
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
# from sklearn.preprocessing import StandardScaler
#
# # Read data from CSV file
# csv_file = '/home/israt/OMNETPP/ts/simu5G/src/data/inputGPR.csv'  # Replace 'your_data.csv' with the path to your CSV file
# data_df = pd.read_csv(csv_file)
#
# # Extract 't', 'x', and 'y' values
# # time_data = data_df['t'].values
# # x_values = data_df['x'].values
# # y_values = data_df['y'].values
#
# # time_data = data_df.iloc[:, 0].values.reshape(-1, 1)
# # x_values = data_df.iloc[:, 1].values
# # y_values = data_df.iloc[:, 2].values
#
# last_xNum_data = data_df[-2000:]
#
# time_data = last_xNum_data.iloc[:, 0].values.reshape(-1, 1)
# x_values = last_xNum_data.iloc[:, 1].values
# y_values = last_xNum_data.iloc[:, 2].values
#
# # Normalize features if necessary
# scaler = StandardScaler()
# time_data_normalized = scaler.fit_transform(time_data.reshape(-1, 1))
#
# # Combine features
# features = np.column_stack((time_data_normalized, x_values))
#
# # Define kernel
# kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
#
# # Create Gaussian Process model
# gp_x = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
# gp_y = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
#
# # Fit the models
# gp_x.fit(features, x_values)
# gp_y.fit(features, y_values)
#
# # future_time_1 = pd.read_csv('/home/mmayon/VEINS_FRAMEWORK/ProjectVNHOv5/simu5G/src/stack/DataFiles/inputGPR_predTimeCoordXY.csv')
# # print('future_time_1: ', future_time_1)
# with open('/home/israt/OMNETPP/ts/simu5G/src/data/inputGPR_predTimeCoordXY.txt', 'r') as file:
#     # Read each line in the file
#     for line in file:
#         # Split the line into individual elements (assuming integers are separated by spaces)
#         elements = line.split()
#
#         # Iterate over the elements and convert them to integers
#         integers = [int(element) for element in elements]
#
#         # Now you can work with the integers in your code
#         # print(integers)
# future_time_1 = integers[0]
# future_time_2 = future_time_1+1
# future_time_3 = future_time_1+2
#
# # Predict future time points
# future_time_data = np.array([[future_time_1], [future_time_2], [future_time_3]])  # Example future time points
# future_time_data_normalized = scaler.transform(future_time_data)
#
# # Predict future 'x' and 'y' values
# future_features = np.column_stack((future_time_data_normalized, np.zeros(len(future_time_data))))
# future_x_values_pred = gp_x.predict(future_features)
# future_y_values_pred = gp_y.predict(future_features)
#
# # Print predicted future values
# # for t, x, y in zip(future_time_data.flatten(), future_x_values_pred, future_y_values_pred):
# #     print(f"Predicted values at time {t}: x={x}, y={y}")
#
# avgX = sum(future_x_values_pred) / len(future_x_values_pred)
# avgX = int(avgX * 10**12)
# avgX = str(avgX)[:4]
# avgX = int(avgX)
# avgY = sum(future_y_values_pred) / len(future_y_values_pred)
# avgY = int(avgY * 10**12)
# avgY = str(avgY)[:4]
# avgY = int(avgY)
# # print('avgX: ', avgX, ', avgY: ', avgY)
#
# with open(r'/home/israt/OMNETPP/ts/simu5G/src/data/outputGPR_predTimeCoordXY.txt', 'w') as f:
#     f.write(f"{avgX}\t{avgY}")




import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

data_df = pd.read_csv('/home/israt/OMNETPP/ts/simu5G/src/data/inputGPR.csv')
last_xNum_data = data_df[-5000:] if len(data_df) > 5000 else data_df
if len(last_xNum_data) < 10:
    print("0 0")
    exit(0)

time_data = last_xNum_data.iloc[:, 0].values.reshape(-1, 1)
x_values = last_xNum_data.iloc[:, 1].values
y_values = last_xNum_data.iloc[:, 2].values

scaler = StandardScaler()
time_data_normalized = scaler.fit_transform(time_data)

features = np.column_stack((time_data_normalized, x_values))
kernel = C(1.0, (1e-2, 1e4)) * RBF(1.0, (1e-2, 1e3)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e6))

gp_x = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp_y = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp_x.fit(features, x_values)
gp_y.fit(features, y_values)

try:
    with open('/home/israt/OMNETPP/ts/simu5G/src/data/inputGPR_predTimeCoordXY.txt', 'r') as file:
        future_time_1 = int(file.read().strip().split()[0])
except:
    future_time_1 = int(time_data[-1]) + 1
future_time_data = np.array([[future_time_1], [future_time_1 + 1], [future_time_1 + 2]])
future_time_data_normalized = scaler.transform(future_time_data)

future_features = np.column_stack((future_time_data_normalized, np.zeros(3)))
future_x_values_pred = gp_x.predict(future_features)
future_y_values_pred = gp_y.predict(future_features)

avgX = int(sum(future_x_values_pred) / len(future_x_values_pred))
avgY = int(sum(future_y_values_pred) / len(future_y_values_pred))
print(f"{avgX} {avgY}")