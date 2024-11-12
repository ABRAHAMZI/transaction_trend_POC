import streamlit as st
import bcrypt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
from scipy.stats import linregress
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv("cred.env")
secure_password=os.getenv("PASSWORD")
stored_hash = bcrypt.hashpw(secure_password.encode('utf-8'), bcrypt.gensalt())

# Function to handle authentication with password hashing
def authenticate():
    # Create the login form
    username = st.text_input("Username", "")
    password = st.text_input("Password", "", type="password")

    # When the user clicks the "Login" button
    if st.button("Login"):
        # Check if the entered username and password match
        if username == "adam" and bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            st.success("Login successful!")
            return True
        else:
            st.error("Invalid username or password.")
            return False

# LSTM Model and Visualization
def run_lstm_model():
    # Load data
    df = pd.read_csv('synthetic_bank_transactions.csv')

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Create date-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # Apply OneHotEncoder for categorical columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # `drop='first'` avoids dummy variable trap

    # Encode 'sender' and 'transaction_type' separately and then add to df
    encoded_features = encoder.fit_transform(df[['sender', 'transaction_type']])
    encoded_feature_names = encoder.get_feature_names_out(['sender', 'transaction_type'])

    # Create a DataFrame from the encoded features and concatenate with the original DataFrame
    df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
    df = pd.concat([df, df_encoded], axis=1).drop(columns=['sender', 'transaction_type'])

    # Scale numerical features
    scaler = MinMaxScaler()
    df[['transaction_amount', 'balance']] = scaler.fit_transform(df[['transaction_amount', 'balance']])

    # Convert DataFrame to numpy array after preprocessing to simplify indexing
    data = df.drop(columns=['date']).values

    # Prepare sequences function with fixed indexing
    def create_sequences(data, window_size):
        sequences = []
        targets = []
        for i in range(len(data) - window_size):
            # Get the sequence of data for the current window
            sequence = data[i:i + window_size]
            # Get the target value (transaction_amount) at the end of the current window
            target = data[i + window_size, 0]  # Assuming 'transaction_amount' is the first column after drop
            sequences.append(sequence)
            targets.append(target)
        return np.array(sequences), np.array(targets)

    # Define window size
    window_size = 60
    X, y = create_sequences(data, window_size)

    X = np.array(X, dtype=np.float32)  # Ensuring X is a NumPy array of floats
    y = np.array(y, dtype=np.float32)  # Ensuring y is a NumPy array of floats
    num_features = X.shape[2]
    X = X.reshape((X.shape[0], X.shape[1], num_features))

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(window_size, num_features)))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Predicting one value (next transaction amount)

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X, y, epochs=30, batch_size=180, verbose=1)

    # Number of future predictions you want
    n_future = 90  # Predicting the next 90 transactions, for example

    # Initialize an array to store future predictions
    future_predictions = []

    # Take the last sequence from the training data to start predicting future values
    current_sequence = X[-1]  # Last sequence from X

    for _ in range(n_future):
        # Predict the next value
        next_value = model.predict(current_sequence.reshape(1, window_size, num_features))[0, 0]
        future_predictions.append(next_value)

        # Create a placeholder for the next sequence
        next_sequence = np.append(current_sequence[1:], [[next_value] + [0] * (num_features - 1)], axis=0)

        # Update the current sequence
        current_sequence = next_sequence

    # Reshape future_predictions for inverse scaling
    future_predictions = np.array(future_predictions).reshape(-1, 1)

    # Create an array with the same shape as the scaler's input
    # Fill it with zeros for columns other than 'transaction_amount'
    filler_array = np.zeros((future_predictions.shape[0], 2))  # Assuming two columns: 'transaction_amount' and 'balance'
    filler_array[:, 0] = future_predictions[:, 0]  # Place predictions in the 'transaction_amount' column

    # Perform inverse scaling
    future_predictions_rescaled = scaler.inverse_transform(filler_array)

    # Extract the rescaled 'transaction_amount' column only
    future_transaction_amounts = future_predictions_rescaled[:, 0]

    # Ensure y_values only contains 90 values (one for each future prediction)
    y_values = future_transaction_amounts.flatten()[:n_future]  # Ensure we use only n_future values
    x_values = np.arange(n_future)  # Days or sequence indices

    # Calculate slope for trend analysis
    slope, intercept, _, _, _ = linregress(x_values, y_values)
    trend = "Positive" if slope > 0 else "Negative"

    st.write(f"The trend of future transactions is: {trend}")

    # Plot future predictions
    plt.plot(y_values, label="Future Predictions")
    plt.title(f"Future Transaction Trend: {trend}")
    plt.xlabel("Future Transactions")
    plt.ylabel("Transaction Amount")
    plt.legend()
    st.pyplot(plt)

# Main code to run the Streamlit app
def main():
    st.title("Welcome to the Streamlit App!")

    # Show the login form and authenticate
    if authenticate():
        # If authentication is successful, show the rest of the app
        st.write("You are now logged in.")

        # Run the LSTM model and show the predictions
        run_lstm_model()

# Run the Streamlit app
if __name__ == "__main__":
    main()
