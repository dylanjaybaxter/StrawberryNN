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
daily_MAX_EPOCHS = 100
daily_CONV_WIDTH = 1
daily_PATIENCE = 10

# Window Parameters
daily_INPUT_WIDTH = 3
daily_LABEL_WIDTH = 1
daily_SHIFT = 1
daily_BATCH_SIZE = 32
daily_STRIDE = 1
daily_SHUFFLE = True

# Model Parameters (Type, Param1(Usually number of nodes))
daily_LAYERS = [['LSTM', 64], ['Dropout', 0.2],['LSTM', 64], ['Dropout', 0.2],['Flatten', 0], ['Dense', 32],['Dense', 1]]


# --------------------------------------------------Data Aquisition-------------------------------------------------
daily_filepath = "C:\\Users\\Dylan\OneDrive - California Polytechnic State University\\Summer 2020\\Strawberry Data\\Strawberry Commission Data\\Summaries\\"
daily_filename = "Daily_SANTA MARIA_FullDataset_WEATHER.csv"# "Daily_Diff_Interp_14_SANTA MARIA_FullDataset_WEATHER.csv"
daily_data = pd.read_csv(daily_filepath + daily_filename)
daily_timestamps = pd.to_datetime(daily_data.pop('Unnamed: 0'), format='%Y-%m-%d')
print(daily_data.describe().transpose())

# --------------------------------------------------Preprocessing-------------------------------------------------
# convert time to sinusoidal signals
daily_test = daily_timestamps[0:]
daily_data_time_day = daily_test.dt.day
daily_data_time_month = daily_test.dt.month
daily_data_time_year = daily_test.dt.year

daily_data['Day sin'] = np.sin(daily_data_time_day * (2 * np.pi / 31))
daily_data['Day cos'] = np.cos(daily_data_time_day * (2 * np.pi / 31))
daily_data['Month sin'] = np.sin(daily_data_time_month * (2 * np.pi / 12))
daily_data['Month cos'] = np.cos(daily_data_time_month * (2 * np.pi / 12))
daily_data['Year'] = daily_data_time_year
daily_data['Day of Year'] = daily_test.dt.dayofyear

#Drop TSUN Due to Nan column
daily_data = daily_data.drop('TSUN', axis='columns')


print(daily_data)

# --------------------------------------------------Splitting-------------------------------------------------

daily_column_indices = {name: i for i, name in enumerate(daily_data.columns)}

daily_n = len(daily_data)
daily_DB_low = int(daily_n * 0.7)
daily_DB_high = int(daily_n * 0.9)
daily_trainingData = daily_data[0:daily_DB_low]
daily_validationData = daily_data[daily_DB_low:daily_DB_high]
daily_testingData = daily_data[daily_DB_high:daily_n]

# --------------------------------------------------Normalization-------------------------------------------------

daily_mean_matrix = daily_trainingData.mean()
daily_stdDev = daily_trainingData.std()
daily_data.describe().transpose()

daily_trainingData = (daily_trainingData - daily_mean_matrix) / daily_stdDev
daily_validationData = (daily_validationData - daily_mean_matrix) / daily_stdDev
daily_testingData = (daily_testingData - daily_mean_matrix) / daily_stdDev

print(0)


# --------------------------------------------------Windowing-------------------------------------------------
window = TrainingWindow(daily_INPUT_WIDTH, daily_LABEL_WIDTH, daily_SHIFT,
                        daily_trainingData, daily_validationData, daily_testingData,
                        label_col_names = ['Trays/Acre'])

# --------------------------------------------------Modeling-------------------------------------------------
daily_lstm_model = tf.keras.models.Sequential()
for layer in daily_LAYERS:
    if layer[0] == 'LSTM':
        daily_lstm_model.add(tf.keras.layers.LSTM(layer[1], return_sequences=True))

    elif layer[0] == 'Dropout':
        daily_lstm_model.add(tf.keras.layers.Dropout(layer[1]))

    elif layer[0] == 'Flatten':
        daily_lstm_model.add(tf.keras.layers.Flatten())

    elif layer[0] == 'Dense':
        daily_lstm_model.add(tf.keras.layers.Dense(units=layer[1]))

# --------------------------------------------------Compile and Fit-------------------------------------------------

daily_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=daily_PATIENCE,
                                                      mode='min',
                                                      restore_best_weights=True)
daily_lstm_model.compile(loss=tf.losses.MeanSquaredError(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanSquaredError()])

daily_train_ds = window.make_dataset(window.training_df, daily_STRIDE, daily_SHUFFLE, daily_BATCH_SIZE)
daily_val_ds = window.make_dataset(window.validation_df, daily_STRIDE, daily_SHUFFLE, daily_BATCH_SIZE)
daily_test_ds = window.make_dataset(window.testing_df, daily_STRIDE, daily_SHUFFLE, daily_BATCH_SIZE)

history = daily_lstm_model.fit(daily_train_ds,
                         epochs=daily_MAX_EPOCHS,
                         validation_data= daily_val_ds,
                         callbacks=[daily_early_stopping])

daily_lstm_model.summary()

# --------------------------------------------------Prediction and Metrics----------------------------------------------
# Metrics (list: loss, MSE)
# loss =
daily_train_performance = daily_lstm_model.evaluate(daily_train_ds)
daily_val_performance = daily_lstm_model.evaluate(daily_val_ds)
daily_test_performance = daily_lstm_model.evaluate(daily_test_ds)

print('')
print('Training: ')
print(daily_lstm_model.metrics_names)
print(daily_train_performance)
print('Validation:')
print(daily_lstm_model.metrics_names)
print(daily_val_performance)
print('Testing: ')
print(daily_lstm_model.metrics_names)
print(daily_test_performance)

# Predictions and Truth Data
daily_truth = np.array(daily_validationData['Trays/Acre'])
daily_predictions = np.array(0)
daily_predictionStart = daily_INPUT_WIDTH + daily_SHIFT - daily_LABEL_WIDTH
daily_length = daily_validationData['Trays/Acre'].count() - daily_predictionStart
for i in range(daily_length):
    inputer = np.array([daily_validationData.iloc[i: i + daily_INPUT_WIDTH].values])
    prediction = daily_lstm_model.predict(inputer)
    daily_predictions = np.append(daily_predictions, prediction)
daily_predictions = daily_predictions[1:]

# Calculate Mean Squared Error
daily_diff_val = daily_predictions - daily_truth[daily_predictionStart:]
daily_diffSquare = np.multiply(daily_diff_val, daily_diff_val)
daily_MSE_Val = np.mean(daily_diffSquare)
print('MSE CALC: ')
print(daily_MSE_Val)

daily_diff_denormed_val = (daily_diff_val * daily_stdDev['Trays/Acre'])

# ---------------------------------------------Excel Export----------------------------------------------
# Get Filename From User
print("Enter Model Information/Filename('Y' for auto filename): ")
filename = str(input())
if filename == 'Y':
    filename = 'LSTM_'
    for layer in daily_LAYERS:
        if layer[0] == 'LSTM':
            filename = filename + 'L' + str(layer[1]) + '_'

        elif layer[0] == 'Dropout':
            filename = filename + 'Dr' + str(layer[1]) + '_'

        elif layer[0] == 'Flatten':
            filename = filename + 'F' + str(layer[1]) + '_'

        elif layer[0] == 'Dense':
            filename = filename + 'D' + str(layer[1]) + '_'

    filename = filename + 'Bt' + str(daily_BATCH_SIZE) + '_'
    filename = filename + 'P' + str(daily_PATIENCE) + '_'
    filename = filename + 'In' + str(daily_INPUT_WIDTH) + '_'
    filename = filename + 'Sh' + str(daily_SHIFT)

filename = filename + '.xlsx'
filepath = 'C:\\Users\\Dylan\OneDrive - California Polytechnic State University\\Summer 2020\\Strawberry Data\\Strawberry Commission Data\\Neural Network Results\\'
sheetName = 'Data'


# Prepare Data
daily_dataOut = pd.DataFrame()
daily_dataOut['Timestamp'] = daily_timestamps[(int(daily_n * 0.7)):int(daily_n * 0.9)]
daily_dataOut['Truth Data'] = daily_truth * daily_stdDev['Trays/Acre'] + daily_mean_matrix['Trays/Acre']
daily_dataOut['Predictions'] = np.concatenate([np.zeros(daily_predictionStart), daily_predictions]) * daily_stdDev['Trays/Acre'] + daily_mean_matrix['Trays/Acre']
daily_dataOut['Difference'] = np.concatenate([np.zeros(daily_predictionStart), daily_diff_denormed_val])

# Write Dataframe to File
writer = pd.ExcelWriter(filepath + filename, engine='xlsxwriter')
daily_dataOut.to_excel(writer, sheet_name='Validation Data', index=False)
workbook = writer.book
worksheet = writer.sheets['Validation Data']

# Chart Prediction vs Truth
chart = workbook.add_chart({'type': 'line'})
for i in range(2):
    chart.add_series({
        'name': ['Validation Data', 0, i + 1],
        'catagories': ['Validation Data', 1, 0, len(daily_truth), 0],
        'values': ['Validation Data', 1, i + 1, len(daily_truth), i + 1],
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
    'catagories': ['Validation Data', 1, 0, len(daily_truth), 0],
    'values': ['Validation Data', 1, 3, len(daily_truth), 3],
})
chart2.set_title({'name': 'Difference'})
chart2.set_x_axis({'name': 'Time',
                   'date_axis': True, })
chart2.set_y_axis({'name': 'Trays/Acre'})
worksheet.insert_chart('G30', chart2)

# --------------------------------------------Test Data---------------------------------------------------
# Predictions and Truth Data
daily_truth = np.array(daily_testingData['Trays/Acre'])
predictions = np.array(0)
predictionStart = daily_INPUT_WIDTH + daily_SHIFT - daily_LABEL_WIDTH
length = daily_testingData['Trays/Acre'].count() - predictionStart
for i in range(length):
    inputer = np.array([daily_testingData.iloc[i: i + daily_INPUT_WIDTH].values])
    prediction = daily_lstm_model.predict(inputer)
    predictions = np.append(predictions, prediction)
predictions = predictions[1:]

# Calculate Mean Squared Error
diff_test = predictions - daily_truth[predictionStart:]
diffSquare = np.multiply(diff_test, diff_test)
MSE_Test = np.mean(diffSquare)
print('MSE CALC: ')
print(MSE_Test)

diff_denormed_test = (diff_test * daily_stdDev['Trays/Acre'])

# ---------------------------------------------Excel Export----------------------------------------------

# Prepare Data
dataOut_test = pd.DataFrame()
dataOut_test['Timestamp'] = daily_timestamps[(int(daily_n * 0.9)):]
dataOut_test['Truth Data'] = daily_truth * daily_stdDev['Trays/Acre'] + daily_mean_matrix['Trays/Acre']
dataOut_test['Predictions'] = np.concatenate([np.zeros(predictionStart), predictions]) * daily_stdDev['Trays/Acre'] + daily_mean_matrix['Trays/Acre']
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
        'catagories': ['Test Data', 1, 0, len(daily_truth), 0],
        'values': ['Test Data', 1, i + 1, len(daily_truth), i + 1],
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
    'catagories': ['Test Data', 1, 0, len(daily_truth), 0],
    'values': ['Test Data', 1, 3, len(daily_truth), 3],
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
worksheet2.write(1, 1, daily_val_performance[1])
worksheet2.write(1, 2, (daily_val_performance[1] * daily_stdDev['Trays/Acre']) + daily_mean_matrix['Trays/Acre'])
worksheet2.write(2, 0, 'Calc Val MSE')
worksheet2.write(2, 1, daily_MSE_Val)
worksheet2.write(2, 2, (daily_MSE_Val * daily_stdDev['Trays/Acre']) + daily_mean_matrix['Trays/Acre'])
worksheet2.write(3, 0, 'Test MSE')
worksheet2.write(3, 1, daily_test_performance[1])
worksheet2.write(3, 2, (daily_test_performance[1] * daily_stdDev['Trays/Acre']) + daily_mean_matrix['Trays/Acre'])
worksheet2.write(4, 0, 'Calc Test MSE')
worksheet2.write(4, 1, MSE_Test)
worksheet2.write(4, 2, (MSE_Test * daily_stdDev['Trays/Acre']) + daily_mean_matrix['Trays/Acre'])
worksheet2.write(5, 0, 'Max Diff Val')
worksheet2.write(5, 1, max(daily_diff_val))
worksheet2.write(5, 2, (max(daily_diff_val) * daily_stdDev['Trays/Acre']) + daily_mean_matrix['Trays/Acre'])
worksheet2.write(6, 0, 'Max Diff Test')
worksheet2.write(6, 1, max(diff_test))
worksheet2.write(6, 2, (max(diff_test) * daily_stdDev['Trays/Acre']) + daily_mean_matrix['Trays/Acre'])

writer.save()

print("Finished")