import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from keras.callbacks import EarlyStopping
from keras.utils import Sequence
from keras.optimizers import Adam
from keras.regularizers import l2
import seaborn as sns

# fix random seed for reproducibility
np.random.seed(7)

data_path = '/kaggle/input/cryptodata/xrdusd.csv'
df = pd.read_csv(data_path, index_col='time')
df.index = pd.to_datetime(df.index, unit='ms')
df = df[~df.index.duplicated(keep='first')]
df = df.resample('min').ffill()

look_back = 2
rsi = RSIIndicator(df.close, window=look_back).rsi()
sma = df.close.rolling(window=look_back).mean()
ema = df.close.ewm(span=look_back, adjust=False).mean()
df['rsi'] = rsi
df['sma'] = sma
df['ema'] = ema

df.drop(['Unnamed: 0'], axis=1, inplace=True)
# Ensure there are no NaN values
df = df.dropna()


df.head(5)

plt.figure(figsize=(15,10))
plt.plot(df.index[:], df.close[:])
plt.show()

def create_dataset(dataset, look_back=14):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

xrpdata = df.iloc[:].values
xrpdata = xrpdata.astype('float32')

scaler = StandardScaler()
xrpdata = scaler.fit_transform(xrpdata)
X, y = create_dataset(xrpdata, look_back)

# Split into training, validation, and test sets
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)
test_size = len(X) - train_size - val_size

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

# Define a data generator to optimize memory usage
class DataGenerator(Sequence):
    def __init__(self, dataX, dataY, batch_size):
        self.dataX = dataX
        self.dataY = dataY
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.dataX) / self.batch_size))

    def __getitem__(self, index):
        batchX = self.dataX[index * self.batch_size:(index + 1) * self.batch_size]
        batchY = self.dataY[index * self.batch_size:(index + 1) * self.batch_size]
        return batchX, batchY
    
# Function to invert transformation for multi-feature predictions
def invert_scaling(predictions, scaler, n_features):
    temp_array = np.zeros((len(predictions), n_features))
    temp_array[:, 0] = predictions.flatten()
    inverted_predictions = scaler.inverse_transform(temp_array)
    return inverted_predictions[:, 0]

# Define the models with L2 regularization and Bidirectional layers
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    model.add(LSTM(50, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(25, kernel_regularizer=l2(0.01)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return model

def create_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    model.add(GRU(50, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(25, kernel_regularizer=l2(0.01)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return model
	
	# Use the generator for training
batch_size = 32
train_generator = DataGenerator(X_train, y_train, batch_size)
val_generator = DataGenerator(X_val, y_val, batch_size)
test_generator = DataGenerator(X_test, y_test, batch_size)

models = {
    'Linear Regression': LinearRegression(),
    'LSTM': create_lstm_model((look_back, X_train.shape[2])),
    'GRU': create_gru_model((look_back, X_train.shape[2])),
}

results = {}
train_predictions = {}
test_predictions = {}

for name, model in models.items():
    print(f"Training {name}...")
    if name in ['LSTM', 'GRU']:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(train_generator, epochs=30, validation_data=val_generator, verbose=2, callbacks=[early_stopping])
        trainPredict = model.predict(train_generator)
        testPredict = model.predict(test_generator)
    else:
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        trainPredict = model.predict(X_train.reshape(X_train.shape[0], -1))
        valPredict = model.predict(X_val.reshape(X_val.shape[0], -1))
        testPredict = model.predict(X_test.reshape(X_test.shape[0], -1))
    
        # Calculate RMSE for validation set
        valPredict = invert_scaling(valPredict, scaler, X_val.shape[2])
        valY_inv = invert_scaling(y_val.reshape(-1, 1), scaler, X_val.shape[2])
        valY_flat = valY_inv[:len(valPredict)].flatten()
        valPredict_flat = valPredict.flatten()
        val_rmse = np.sqrt(mean_squared_error(valY_flat, valPredict_flat))

        if name not in results:
            results[name] = {
                'train_rmse': [],
                'val_rmse': [],
                'test_rmse': []
            }
        results[name]['val_rmse'].append(val_rmse)

    # Invert predictions and actual values for scaling back to the original range
    trainPredict = invert_scaling(trainPredict, scaler, X_train.shape[2])
    testPredict = invert_scaling(testPredict, scaler, X_test.shape[2])
    trainY_inv = invert_scaling(y_train.reshape(-1, 1), scaler, X_train.shape[2])
    testY_inv = invert_scaling(y_test.reshape(-1, 1), scaler, X_test.shape[2])

    # Flatten the arrays for computing RMSE
    trainY_flat = trainY_inv[:len(trainPredict)].flatten()
    testY_flat = testY_inv[:len(testPredict)].flatten()
    trainPredict_flat = trainPredict.flatten()
    testPredict_flat = testPredict.flatten()

    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(trainY_flat, trainPredict_flat))
    test_rmse = np.sqrt(mean_squared_error(testY_flat, testPredict_flat))
    
    if name not in results:
        results[name] = {
            'train_rmse': [],
            'test_rmse': []
        }
    results[name]['train_rmse'].append(train_rmse)
    results[name]['test_rmse'].append(test_rmse)

    if name not in train_predictions:
        train_predictions[name] = []
    if name not in test_predictions:
        test_predictions[name] = []

    train_predictions[name].append(trainPredict_flat)
    test_predictions[name].append(testPredict_flat)

# Print the results
for name, metrics in results.items():
    print(f"{name}: Train RMSE = {metrics['train_rmse'][0]}, Test RMSE = {metrics['test_rmse'][0]}")
    if 'val_rmse' in metrics:
        print(f"{name}: Val RMSE = {metrics['val_rmse'][0]}")
		
# RMSE Comparison Bar Chart
model_names = list(results.keys())
train_rmse = [results[name]['train_rmse'] for name in model_names]
test_rmse = [results[name]['test_rmse'] for name in model_names]

plt.figure(figsize=(12, 6))
x = range(len(model_names))
plt.bar(x, train_rmse, width=0.4, label='Train RMSE', align='center')
plt.bar(x, test_rmse, width=0.4, label='Test RMSE', align='edge')
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.xticks(x, model_names, rotation=45)
plt.title('RMSE Comparison of Different Models')
plt.legend()
plt.savefig('/kaggle/working/rmse_comparison.png')
plt.show()

# Prediction vs Actual Plot
plt.figure(figsize=(15, 10))
plt.plot(testY_flat, label='Actual', color='black')
for name in model_names:
    plt.plot(test_predictions[name], label=f'{name} Predictions')
plt.title(f'Actual vs Predictions')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig('/kaggle/working/test_predict_plot.png')
plt.show()