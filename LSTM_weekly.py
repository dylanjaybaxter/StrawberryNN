import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from TrainingWindow import *

# Seed Randomization
np.random.seed(1)
tf.random.set_seed(2)

# CUDAS Setup
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
# --------------------------------------------------Parameters-------------------------------------------------

# Training Parameters
MAX_EPOCHS = 50
CONV_WIDTH = 1
PATIENCE = 4

# Window Parameters
INPUT_WIDTH = 3
LABEL_WIDTH = 1
SHIFT = 1
BATCH_SIZE = 32
STRIDE = 1
SHUFFLE = True

# Model Parameters (Type, Param1(Usually number of nodes))
LAYERS = [['LSTM', 64], ['Dropout', 0.2],['LSTM', 64], ['Dropout', 0.2],['Flatten', 0], ['Dense', 32],['Dense', 1]]


# --------------------------------------------------Data Aquisition-------------------------------------------------
filepath = "C:\\Users\\Dylan\OneDrive - California Polytechnic State University\\Summer 2020\\Strawberry Data\\Strawberry Commission Data\\Summaries\\"
filename = "Weekly_StrawberryCommission_SANTA MARIA_FullDataset_Weather.csv"  # "Daily_SANTA MARIA_FullDataset_WEATHER.csv"# "Daily_Diff_Interp_14_SANTA MARIA_FullDataset_WEATHER.csv"
data = pd.read_csv(filepath + filename)
timestamps = pd.to_datetime(data.pop('Week Ending'), format='%m/%d/%Y')
print(data.describe().transpose())

# --------------------------------------------------Preprocessing-------------------------------------------------
# convert time to sinusoidal signals
test = timestamps[0:]
data_time_day = test.dt.day
data_time_month = test.dt.month
data_time_year = test.dt.year

data['Day sin'] = np.sin(data_time_day * (2 * np.pi / 31))
data['Day cos'] = np.cos(data_time_day * (2 * np.pi / 31))
data['Month sin'] = np.sin(data_time_month * (2 * np.pi / 12))
data['Month cos'] = np.cos(data_time_month * (2 * np.pi / 12))
data['Year'] = data_time_year
data['Day of Year'] = test.dt.dayofyear

print(data)

# --------------------------------------------------Splitting-------------------------------------------------

column_indices = {name: i for i, name in enumerate(data.columns)}

n = len(data)
DB_low = int(n * 0.7)
DB_high = int(n * 0.9)
trainingData = data[0:DB_low]
validationData = data[DB_low:DB_high]
testingData = data[DB_high:n]

# --------------------------------------------------Normalization-------------------------------------------------

mean_matrix = trainingData.mean()
stdDev = trainingData.std()
data.describe().transpose()

trainingData = (trainingData - mean_matrix) / stdDev
validationData = (validationData - mean_matrix) / stdDev
testingData = (testingData - mean_matrix) / stdDev

print(0)


# --------------------------------------------------Windowing-------------------------------------------------
window = TrainingWindow(INPUT_WIDTH, LABEL_WIDTH, SHIFT,
                        trainingData, validationData, testingData,
                        label_col_names = ['Trays/Acre'])

# --------------------------------------------------Modeling-------------------------------------------------
lstm_model = tf.keras.models.Sequential()
for layer in LAYERS:
    if layer[0] == 'LSTM':
        lstm_model.add(tf.keras.layers.LSTM(layer[1], return_sequences=True))

    elif layer[0] == 'Dropout':
        lstm_model.add(tf.keras.layers.Dropout(layer[1]))

    elif layer[0] == 'Flatten':
        lstm_model.add(tf.keras.layers.Flatten())

    elif layer[0] == 'Dense':
        lstm_model.add(tf.keras.layers.Dense(units=layer[1]))

# --------------------------------------------------Compile and Fit-------------------------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=PATIENCE,
                                                      mode='min',
                                                      restore_best_weights=True)
lstm_model.compile(loss=tf.losses.MeanSquaredError(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanSquaredError()])

train_ds = window.make_dataset(window.training_df, STRIDE, SHUFFLE, BATCH_SIZE)
val_ds = window.make_dataset(window.validation_df, STRIDE, SHUFFLE, BATCH_SIZE)
test_ds = window.make_dataset(window.testing_df, STRIDE, SHUFFLE, BATCH_SIZE)

history = lstm_model.fit(train_ds,
                         epochs=MAX_EPOCHS,
                         validation_data= val_ds,
                         callbacks=[early_stopping])

lstm_model.summary()

# --------------------------------------------------Prediction and Metrics----------------------------------------------
# Metrics (list: loss, MSE)
# loss =
train_performance = lstm_model.evaluate(train_ds)
val_performance = lstm_model.evaluate(val_ds)
test_performance = lstm_model.evaluate(test_ds)

print('')
print('Training: ')
print(lstm_model.metrics_names)
print(train_performance)
print('Validation:')
print(lstm_model.metrics_names)
print(val_performance)
print('Testing: ')
print(lstm_model.metrics_names)
print(test_performance)

# Predictions and Truth Data
truth = np.array(validationData['Trays/Acre'])
predictions = np.array(0)
predictionStart = INPUT_WIDTH + SHIFT - LABEL_WIDTH
length = validationData['Trays/Acre'].count() - predictionStart
for i in range(length):
    inputer = np.array([validationData.iloc[i: i + INPUT_WIDTH].values])
    prediction = lstm_model.predict(inputer)
    predictions = np.append(predictions, prediction)
predictions = predictions[1:]

# Calculate Mean Squared Error
diff_val = predictions - truth[predictionStart:]
diffSquare = np.multiply(diff_val, diff_val)
MSE_Val = np.mean(diffSquare)
print('MSE CALC: ')
print(MSE_Val)

diff_denormed_val = (diff_val * stdDev['Trays/Acre'])

# ---------------------------------------------Excel Export----------------------------------------------
# Get Filename From User
print("Enter Model Information/Filename('Y' for auto filename): ")
filename = str(input())
if filename == 'Y':
    filename = 'LSTM_'
    for layer in LAYERS:
        if layer[0] == 'LSTM':
            filename = filename + 'L' + str(layer[1]) + '_'

        elif layer[0] == 'Dropout':
            filename = filename + 'Dr' + str(layer[1]) + '_'

        elif layer[0] == 'Flatten':
            filename = filename + 'F' + str(layer[1]) + '_'

        elif layer[0] == 'Dense':
            filename = filename + 'D' + str(layer[1]) + '_'

    filename = filename + 'Bt' + str(BATCH_SIZE) + '_'
    filename = filename + 'P' + str(PATIENCE) + '_'
    filename = filename + 'In' + str(INPUT_WIDTH) + '_'
    filename = filename + 'Sh' + str(SHIFT)

filename = filename + '.xlsx'
filepath = 'C:\\Users\\Dylan\OneDrive - California Polytechnic State University\\Summer 2020\\Strawberry Data\\Strawberry Commission Data\\Neural Network Results\\'
sheetName = 'Data'


# Prepare Data
dataOut = pd.DataFrame()
dataOut['Timestamp'] = timestamps[(int(n * 0.7)):int(n * 0.9)]
dataOut['Truth Data'] = truth * stdDev['Trays/Acre'] + mean_matrix['Trays/Acre']
dataOut['Predictions'] = np.concatenate([np.zeros(predictionStart), predictions]) * stdDev['Trays/Acre'] + mean_matrix['Trays/Acre']
dataOut['Difference'] = np.concatenate([np.zeros(predictionStart), diff_denormed_val])

# Write Dataframe to File
writer = pd.ExcelWriter(filepath + filename, engine='xlsxwriter')
dataOut.to_excel(writer, sheet_name='Validation Data', index=False)
workbook = writer.book
worksheet = writer.sheets['Validation Data']

# Chart Prediction vs Truth
chart = workbook.add_chart({'type': 'line'})
for i in range(2):
    chart.add_series({
        'name': ['Validation Data', 0, i + 1],
        'catagories': ['Validation Data', 1, 0, len(truth), 0],
        'values': ['Validation Data', 1, i + 1, len(truth), i + 1],
    })
chart.set_title({'name': 'Truth & Predictions'})
chart.set_x_axis({'name': 'Time',
                  'date_axis': True, })
chart.set_y_axis({'name': 'Trays/Acre'})
worksheet.insert_chart('G2', chart)

# Chart Residuals
chart2 = workbook.add_chart({'type': 'line'})
chart2.add_series({
    'name': ['Validation Data', 0, 3],
    'catagories': ['Validation Data', 1, 0, len(truth), 0],
    'values': ['Validation Data', 1, 3, len(truth), 3],
})
chart2.set_title({'name': 'Difference'})
chart2.set_x_axis({'name': 'Time',
                   'date_axis': True, })
chart2.set_y_axis({'name': 'Trays/Acre'})
worksheet.insert_chart('G30', chart2)

# --------------------------------------------Test Data---------------------------------------------------
# Predictions and Truth Data
truth = np.array(testingData['Trays/Acre'])
predictions = np.array(0)
predictionStart = INPUT_WIDTH + SHIFT - LABEL_WIDTH
length = testingData['Trays/Acre'].count() - predictionStart
for i in range(length):
    inputer = np.array([testingData.iloc[i: i + INPUT_WIDTH].values])
    prediction = lstm_model.predict(inputer)
    predictions = np.append(predictions, prediction)
predictions = predictions[1:]

# Calculate Mean Squared Error
diff_test = predictions - truth[predictionStart:]
diffSquare = np.multiply(diff_test, diff_test)
MSE_Test = np.mean(diffSquare)
print('MSE CALC: ')
print(MSE_Test)

diff_denormed_test = (diff_test * stdDev['Trays/Acre'])

# ---------------------------------------------Excel Export----------------------------------------------

# Prepare Data
dataOut_test = pd.DataFrame()
dataOut_test['Timestamp'] = timestamps[(int(n * 0.9)):]
dataOut_test['Truth Data'] = truth * stdDev['Trays/Acre'] + mean_matrix['Trays/Acre']
dataOut_test['Predictions'] = np.concatenate([np.zeros(predictionStart), predictions]) * stdDev['Trays/Acre'] + mean_matrix['Trays/Acre']
dataOut_test['Difference'] = np.concatenate([np.zeros(predictionStart), diff_denormed_test])

# Write Dataframe to File
dataOut_test.to_excel(writer, sheet_name='Test Data', index=False)
workbook = writer.book
worksheet = writer.sheets['Test Data']

# Chart Prediction vs Truth
chart = workbook.add_chart({'type': 'line'})
for i in range(2):
    chart.add_series({
        'name': ['Test Data', 0, i + 1],
        'catagories': ['Test Data', 1, 0, len(truth), 0],
        'values': ['Test Data', 1, i + 1, len(truth), i + 1],
    })
chart.set_title({'name': 'Truth & Predictions'})
chart.set_x_axis({'name': 'Time',
                  'date_axis': True, })
chart.set_y_axis({'name': 'Trays/Acre'})
worksheet.insert_chart('G2', chart)

# Chart Residuals
chart2 = workbook.add_chart({'type': 'line'})
chart2.add_series({
    'name': ['Test Data', 0, 3],
    'catagories': ['Test Data', 1, 0, len(truth), 0],
    'values': ['Test Data', 1, 3, len(truth), 3],
})
chart2.set_title({'name': 'Difference'})
chart2.set_x_axis({'name': 'Time',
                   'date_axis': True, })
chart2.set_y_axis({'name': 'Trays/Acre'})
worksheet.insert_chart('G30', chart2)

# Write Metrics to Seperate Sheet
worksheet2 = workbook.add_worksheet('Metrics')
worksheet2.write(0, 1, 'Normalized')
worksheet2.write(0, 2, 'Real')
worksheet2.write(1, 0, 'Val MSE')
worksheet2.write(1, 1, val_performance[1])
worksheet2.write(1, 2, (val_performance[1] * stdDev['Trays/Acre']) + mean_matrix['Trays/Acre'])
worksheet2.write(2, 0, 'Calc Val MSE')
worksheet2.write(2, 1, MSE_Val)
worksheet2.write(2, 2, (MSE_Val * stdDev['Trays/Acre']) + mean_matrix['Trays/Acre'])
worksheet2.write(3, 0, 'Test MSE')
worksheet2.write(3, 1, test_performance[1])
worksheet2.write(3, 2, (test_performance[1] * stdDev['Trays/Acre']) + mean_matrix['Trays/Acre'])
worksheet2.write(4, 0, 'Calc Test MSE')
worksheet2.write(4, 1, MSE_Test)
worksheet2.write(4, 2, (MSE_Test * stdDev['Trays/Acre']) + mean_matrix['Trays/Acre'])
worksheet2.write(5, 0, 'Max Diff Val')
worksheet2.write(5, 1, max(diff_val))
worksheet2.write(5, 2, (max(diff_val) * stdDev['Trays/Acre']) + mean_matrix['Trays/Acre'])
worksheet2.write(6, 0, 'Max Diff Test')
worksheet2.write(6, 1, max(diff_test))
worksheet2.write(6, 2, (max(diff_test) * stdDev['Trays/Acre']) + mean_matrix['Trays/Acre'])

writer.save()

print("Finished")