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
MAX_EPOCHS = 100
CONV_WIDTH = 1
PATIENCE = 10

# Window Parameters
INPUT_WIDTH = 3
LABEL_WIDTH = 1
SHIFT = 1
BATCH_SIZE = 96
STRIDE = 1
SHUFFLE = True

# Model Parameters (Type, Param1(Usually number of nodes))
LAYERS = [['LSTM', 64], ['Dropout', 0.2],['Flatten', 0], ['Dense', 32],['Dense', 1]]
    # Best Model Majority of Quarter [['LSTM', 64], ['Dropout', 0.2],['LSTM', 64], ['Dropout', 0.2],['Flatten', 0], ['Dense', 32],['Dense', 1]]

# Training Parameters
daily_MAX_EPOCHS = MAX_EPOCHS
daily_CONV_WIDTH = CONV_WIDTH
daily_PATIENCE = PATIENCE

# Window Parameters
daily_INPUT_WIDTH = 21
daily_LABEL_WIDTH = 1
daily_SHIFT = 7
daily_BATCH_SIZE = BATCH_SIZE
daily_STRIDE = STRIDE
daily_SHUFFLE = SHUFFLE

# Model Parameters (Type, Param1(Usually number of nodes))
daily_LAYERS = LAYERS


# --------------------------------------------------Data Aquisition-------------------------------------------------
filepath = "C:\\Users\\Dylan\OneDrive - California Polytechnic State University\\Summer 2020\\Strawberry Data\\Strawberry Commission Data\\Summaries\\"
filename = "Weekly_StrawberryCommission_SANTA MARIA_FullDataset_Weather.csv"  # "Daily_SANTA MARIA_FullDataset_WEATHER.csv"# "Daily_Diff_Interp_14_SANTA MARIA_FullDataset_WEATHER.csv"
data = pd.read_csv(filepath + filename)
timestamps = pd.to_datetime(data.pop('Week Ending'), format='%m/%d/%Y')
print(data.describe().transpose())

# Get Filename From User
print("Enter Model Information/Filename('Y' for auto filename): ")
filename = str(input())
if filename == 'Y':
    filename = 'LSTM_DvW_'
    for layer in LAYERS:
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
    filename = filename + 'E' + str(daily_MAX_EPOCHS) + '_'
    filename = filename + 'In' + str(daily_INPUT_WIDTH) + '_'
    filename = filename + 'Sh' + str(daily_SHIFT)

filename = filename + '.xlsx'
filepath = 'C:\\Users\\Dylan\OneDrive - California Polytechnic State University\\Summer 2020\\Strawberry Data\\Strawberry Commission Data\\Neural Network Results\\'

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
test_truth = np.array(testingData['Trays/Acre'])
test_predictions = np.array(0)
test_predictionStart = INPUT_WIDTH + SHIFT - LABEL_WIDTH
length = testingData['Trays/Acre'].count() - test_predictionStart
for i in range(length):
    inputer = np.array([testingData.iloc[i: i + INPUT_WIDTH].values])
    prediction = lstm_model.predict(inputer)
    test_predictions = np.append(test_predictions, prediction)
test_predictions = test_predictions[1:]

# Calculate Mean Squared Error
diff_test = test_predictions - test_truth[test_predictionStart:]
diffSquare = np.multiply(diff_test, diff_test)
MSE_Test = np.mean(diffSquare)
print('MSE CALC: ')
print(MSE_Test)

diff_denormed_test = (diff_test * stdDev['Trays/Acre'])

# ---------------------------------------------Excel Export----------------------------------------------

# Prepare Data
dataOut_test = pd.DataFrame()
dataOut_test['Timestamp'] = timestamps[(int(n * 0.9)):]
dataOut_test['Truth Data'] = test_truth * stdDev['Trays/Acre'] + mean_matrix['Trays/Acre']
dataOut_test['Predictions'] = np.concatenate([np.zeros(predictionStart), test_predictions]) * stdDev['Trays/Acre'] + mean_matrix['Trays/Acre']
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


'''-----------------------------------------------------------------------------------------------------------------'''

# Seed Randomization
np.random.seed(1)
tf.random.set_seed(2)


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


daily_to_weekly = []
daily_to_weekly_times = []
zeroes_added = 0
for i in range(DB_low, DB_high+1):
    for j in range(daily_DB_low, daily_DB_high+1):
        if timestamps[i] == daily_timestamps[j]:
            k = j - daily_DB_low - daily_predictionStart
            sum = 0
            while k > j - daily_DB_low - daily_predictionStart - 7 and k > 0:
                sum += daily_predictions[k]
                k -= 1
            if sum == 0:
                zeroes_added += 1
            daily_to_weekly.append(sum*daily_stdDev['Trays/Acre'] + daily_mean_matrix['Trays/Acre']*7)
            daily_to_weekly_times.append(timestamps[i])
while len(daily_to_weekly) < len(predictions):
    daily_to_weekly = [0] + daily_to_weekly
    zeroes_added += 1
daily_to_weekly = (daily_to_weekly - mean_matrix['Trays/Acre'])/stdDev['Trays/Acre']

# Calculate Mean Squared Error
diff_test = daily_to_weekly[zeroes_added:] - truth[predictionStart+zeroes_added:]
diffSquare = np.multiply(diff_test, diff_test)
MSE_Daily_to_Weekly = np.mean(diffSquare)
print('MSE DtW CALC: ')
print(MSE_Daily_to_Weekly)


# ---------------------------------------------Excel Export----------------------------------------------
# Get Filename From User
sheetName = 'Data'

# Prepare Data
daily_dataOut = pd.DataFrame()
daily_dataOut['Timestamp'] = daily_timestamps[(int(daily_n * 0.7)):int(daily_n * 0.9)]
daily_dataOut['Truth Data'] = daily_truth * daily_stdDev['Trays/Acre'] + daily_mean_matrix['Trays/Acre']
daily_dataOut['Predictions'] = np.concatenate([np.zeros(daily_predictionStart), daily_predictions]) * daily_stdDev['Trays/Acre'] + daily_mean_matrix['Trays/Acre']
daily_dataOut['Difference'] = np.concatenate([np.zeros(daily_predictionStart), daily_diff_denormed_val])

# Write Dataframe to File
daily_dataOut.to_excel(writer, sheet_name='Daily Validation Data', index=False)
workbook = writer.book
worksheet = writer.sheets['Daily Validation Data']

# Chart Prediction vs Truth
chart = workbook.add_chart({'type': 'line'})
for i in range(2):
    chart.add_series({
        'name': ['Daily Validation Data', 0, i + 1],
        'catagories': ['Daily Validation Data', 1, 0, len(daily_truth), 0],
        'values': ['Daily Validation Data', 1, i + 1, len(daily_truth), i + 1],
    })
chart.set_title({'name': 'Truth & Predictions'})
chart.set_x_axis({'name': 'Time',
                  'date_axis': True, })
chart.set_y_axis({'name': 'Trays/Acre'})
worksheet.insert_chart('G2', chart)

# Chart Residuals
chart2 = workbook.add_chart({'type': 'line'})
chart2.add_series({
    'name': ['Daily Validation Data', 0, 3],
    'catagories': ['Daily Validation Data', 1, 0, len(daily_truth), 0],
    'values': ['Daily Validation Data', 1, 3, len(daily_truth), 3],
})
chart2.set_title({'name': 'Difference'})
chart2.set_x_axis({'name': 'Time',
                   'date_axis': True, })
chart2.set_y_axis({'name': 'Trays/Acre'})
worksheet.insert_chart('G30', chart2)


# --------------------------------------------Test Data---------------------------------------------------
# Predictions and Truth Data
daily_truth = np.array(daily_testingData['Trays/Acre'])
daily_test_predictions = np.array(0)
daily_test_predictionStart = daily_INPUT_WIDTH + daily_SHIFT - daily_LABEL_WIDTH
length = daily_testingData['Trays/Acre'].count() - daily_test_predictionStart
for i in range(length):
    inputer = np.array([daily_testingData.iloc[i: i + daily_INPUT_WIDTH].values])
    prediction = daily_lstm_model.predict(inputer)
    daily_test_predictions = np.append(daily_test_predictions, prediction)
daily_test_predictions = daily_test_predictions[1:]

# Calculate Mean Squared Error
diff_test = daily_test_predictions - daily_truth[daily_test_predictionStart:]
diffSquare = np.multiply(diff_test, diff_test)
MSE_Test = np.mean(diffSquare)
print('MSE CALC: ')
print(MSE_Test)

diff_denormed_test = (diff_test * daily_stdDev['Trays/Acre'])

daily_to_weekly_test = []
weekly_comparison_test = []
daily_to_weekly_times_test = []
zeroes_added = 0
for i in range(DB_high, n):
    for j in range(daily_DB_high, daily_n):
        if timestamps[i] == daily_timestamps[j]:
            k = j - daily_DB_high - daily_test_predictionStart
            sum = 0
            while k > j - daily_DB_high - daily_test_predictionStart - 7 and k > 0:
                sum += daily_test_predictions[k]
                k -= 1
            if sum == 0:
                zeroes_added += 1
            daily_to_weekly_test.append(sum*daily_stdDev['Trays/Acre'] + daily_mean_matrix['Trays/Acre']*7)
            daily_to_weekly_times_test.append(timestamps[i])
            weekly_comparison_test.append(test_predictions[i-DB_high-test_predictionStart])
while len(daily_to_weekly_test) < len(test_predictions):
    daily_to_weekly_test = [0] + daily_to_weekly_test
    weekly_comparison_test = [0] + weekly_comparison_test
    zeroes_added += 1
daily_to_weekly_test = (daily_to_weekly_test - mean_matrix['Trays/Acre'])/stdDev['Trays/Acre']

# Calculate Mean Squared Error
diff_test = daily_to_weekly_test[zeroes_added:] - test_truth[test_predictionStart+zeroes_added:]
diffSquare = np.multiply(diff_test, diff_test)
MSE_Daily_to_Weekly_test = np.mean(diffSquare)
print('MSE DtW test CALC: ')
print(MSE_Daily_to_Weekly_test)

# ---------------------------------------------Excel Export----------------------------------------------

# Prepare Data
daily_dataOut_test = pd.DataFrame()
daily_dataOut_test['Timestamp'] = daily_timestamps[(int(daily_n * 0.9)):]
daily_dataOut_test['Truth Data'] = daily_truth * daily_stdDev['Trays/Acre'] + daily_mean_matrix['Trays/Acre']
daily_dataOut_test['Predictions'] = np.concatenate([np.zeros(daily_test_predictionStart), daily_test_predictions]) * daily_stdDev['Trays/Acre'] + daily_mean_matrix['Trays/Acre']
daily_dataOut_test['Difference'] = np.concatenate([np.zeros(daily_test_predictionStart), diff_denormed_test])

# Write Dataframe to File
daily_dataOut_test.to_excel(writer, sheet_name='Daily Test Data', index=False)
workbook = writer.book
worksheet = writer.sheets['Daily Test Data']

# Chart Prediction vs Truth
chart = workbook.add_chart({'type': 'line'})
for i in range(2):
    chart.add_series({
        'name': ['Daily Test Data', 0, i + 1],
        'catagories': ['Daily Test Data', 1, 0, len(daily_truth), 0],
        'values': ['Daily Test Data', 1, i + 1, len(daily_truth), i + 1],
    })
chart.set_title({'name': 'Truth & Predictions'})
chart.set_x_axis({'name': 'Time',
                  'date_axis': True, })
chart.set_y_axis({'name': 'Trays/Acre'})
worksheet.insert_chart('G2', chart)

# Chart Residuals
chart2 = workbook.add_chart({'type': 'line'})
chart2.add_series({
    'name': ['Daily Test Data', 0, 3],
    'catagories': ['Daily Test Data', 1, 0, len(daily_truth), 0],
    'values': ['Daily Test Data', 1, 3, len(daily_truth), 3],
})
chart2.set_title({'name': 'Difference'})
chart2.set_x_axis({'name': 'Time',
                   'date_axis': True, })
chart2.set_y_axis({'name': 'Trays/Acre'})
worksheet.insert_chart('G30', chart2)

# Write Metrics to Seperate Sheet
worksheet2 = workbook.add_worksheet('Daily Metrics')
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
worksheet2.write(7, 0, 'Weekly Convert MSE')
worksheet2.write(7, 1, MSE_Daily_to_Weekly)
worksheet2.write(7, 2, (MSE_Daily_to_Weekly * stdDev['Trays/Acre']) + mean_matrix['Trays/Acre'])
worksheet2.write(8, 0, 'Weekly Test Convert MSE')
worksheet2.write(8, 1, MSE_Daily_to_Weekly_test)
worksheet2.write(8, 2, (MSE_Daily_to_Weekly_test * stdDev['Trays/Acre']) + mean_matrix['Trays/Acre'])


# Final Comparisons
dataOut = dataOut.drop('Difference', axis='columns')
dataOut_test = dataOut_test.drop('Difference', axis='columns')
dataOut['Daily Predictions'] = np.concatenate([np.zeros(predictionStart), daily_to_weekly]) * stdDev['Trays/Acre'] + mean_matrix['Trays/Acre']
#dataOut_test['Weekly Predictions Test'] = np.concatenate([np.zeros(predictionStart), weekly_comparison_test])
dataOut_test['Daily Predictions'] = np.concatenate([np.zeros(predictionStart), daily_to_weekly_test]) * stdDev['Trays/Acre'] + mean_matrix['Trays/Acre']



dataOut.to_excel(writer, sheet_name='Val Comparison', index=False)
workbook = writer.book
worksheet_comp = writer.sheets['Val Comparison']

# Chart Prediction vs Truth
chart = workbook.add_chart({'type': 'line'})
for i in range(3):
    chart.add_series({
        'name': ['Val Comparison', 0, i + 1],
        'catagories': ['Val Comparison', 1, 0, len(test_truth), 0],
        'values': ['Val Comparison', 1, i + 1, len(test_truth), i + 1],
    })
chart.set_title({'name': 'Truth & Predictions'})
chart.set_x_axis({'name': 'Time',
                  'date_axis': True, })
chart.set_y_axis({'name': 'Trays/Acre'})
worksheet_comp.insert_chart('G2', chart)

dataOut_test.to_excel(writer, sheet_name='Test Comparison', index=False)
workbook = writer.book
worksheet_comp = writer.sheets['Test Comparison']

# Chart Prediction vs Truth
chart = workbook.add_chart({'type': 'line'})
for i in range(3):
    chart.add_series({
        'name': ['Test Comparison', 0, i + 1],
        'catagories': ['Test Comparison', 1, 0, len(test_truth), 0],
        'values': ['Test Comparison', 1, i + 1, len(test_truth), i + 1],
    })
chart.set_title({'name': 'Truth & Predictions'})
chart.set_x_axis({'name': 'Time',
                  'date_axis': True, })
chart.set_y_axis({'name': 'Trays/Acre'})
worksheet_comp.insert_chart('G2', chart)

writer.save()

print("Finished")