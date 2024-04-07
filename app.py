from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to load and preprocess the data and train the LSTM model
def train_lstm_model(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    data.set_index('Date', inplace=True)

    # Feature scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])

    # Define input sequences and target values
    X, y = [], []
    look_back = 2  # Number of previous days to consider

    for i in range(look_back, len(data)):
        X.append(scaled_data[i - look_back:i])
        y.append(1 if data['Close'][i] > data['Open'][i] else 0)

    X, y = np.array(X), np.array(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=80)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=64)

    return model, X_test, scaler

# Function to make predictions using the trained model
def predict_signals(model, X_test, scaler):
    # Make predictions
    predictions = model.predict(X_test)

    # Output buy/sell signals with probabilities
    results = []
    for i in range(10):
        if predictions[i] > 0.5:
            results.append(f"Day {i+1}: Buy Signal (Probability: {predictions[i][0]:.4f})")
        else:
            results.append(f"Day {i+1}: Sell Signal (Probability: {1 - predictions[i][0]:.4f})")

    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        uploaded_file = request.files['file']
        file_path = f"{app.config['UPLOAD_FOLDER']}/{uploaded_file.filename}"
        uploaded_file.save(file_path)

        # Train the LSTM model and get the test data
        lstm_model, X_test, scaler = train_lstm_model(file_path)

        # Use the trained model to make predictions
        results = predict_signals(lstm_model, X_test, scaler)

        # Render results on the same page or a new page
        return render_template('results.html', results=results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


# import os
# from flask import Flask, render_template, request

# app = Flask(__name__)

# # Set the upload folder
# UPLOAD_FOLDER = 'files'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Handle file upload
#         uploaded_file = request.files['file']
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_file.csv')
#         uploaded_file.save(file_path)

#         # Process the CSV file and generate trading signals
#         # Implement this part based on your stock data processing logic

#         # Example: Assume buy_signal and sell_signal are obtained
#         buy_signal = ["Day 1: Buy Signal", "Day 2: Buy Signal"]
#         sell_signal = ["Day 1: Sell Signal", "Day 2: Sell Signal"]

#         # Render results on the same page or a new page
#         return render_template('results.html', buy_signal=buy_signal, sell_signal=sell_signal)

#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)
