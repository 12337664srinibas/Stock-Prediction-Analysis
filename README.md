# Stock-Prediction-Analysis

Stock Price Prediction Using LSTM
This project uses Long Short-Term Memory (LSTM), a type of recurrent neural network, to predict stock prices based on historical data. The model is built using Python, TensorFlow, and Keras, with stock data sourced from Yahoo Finance using the yfinance library.

Project Overview
Predicting stock prices is one of the most popular applications of data science and machine learning. This project demonstrates how to use an LSTM model to predict future stock prices by training it on historical stock data. The model is trained using the past 60 days' stock prices to predict the next day’s closing price.

Features
Data Source: Stock price data is downloaded using the yfinance library.
Data Preprocessing: The stock prices are scaled between 0 and 1 using MinMaxScaler to normalize the input for the LSTM model.
Model Architecture: The model is built with multiple LSTM layers, followed by Dense layers to output the predicted stock price.
Visualization: Actual vs predicted stock prices are plotted to visualize the model’s performance.
Evaluation: The model’s performance is evaluated using the Root Mean Squared Error (RMSE).
Libraries and Tools Used
Python: Programming language used for the project.
TensorFlow & Keras: To build and train the LSTM model.
NumPy: For numerical operations.
Pandas: For data manipulation and analysis.
Matplotlib: For plotting and visualizing the stock price data.
scikit-learn: For scaling the data using MinMaxScaler.
yfinance: To download historical stock data from Yahoo Finance.
Dataset
We use the stock data of Apple Inc. (AAPL) from 2010 to 2023, sourced from Yahoo Finance. You can replace the stock symbol AAPL in the code with any other stock symbol to predict prices for a different company.

Installation
To run the project, first install the necessary dependencies:



pip install numpy pandas matplotlib scikit-learn tensorflow yfinance
Running the Project
Clone the repository or download the project files.
Open the Jupyter Notebook and run the cells in sequence.
You can change the stock symbol in the yfinance download function to predict prices for a different company.
The model will train on 80% of the data and test on the remaining 20%.
The results (actual vs predicted prices) will be plotted, and the RMSE will be displayed.
Code Structure
Step 1: Import libraries and download stock data.
Step 2: Preprocess the data, normalize it using MinMaxScaler, and split it into training and testing sets.
Step 3: Build the LSTM model with two LSTM layers and two Dense layers.
Step 4: Train the model on the training data.
Step 5: Make predictions on the testing data.
Step 6: Plot the actual vs predicted stock prices and calculate the RMSE to evaluate the model's performance.
Usage
Change Stock Symbol: You can modify the stock symbol from AAPL to any other company’s stock ticker (e.g., GOOGL, MSFT, TSLA).
Adjust Training Period: The default period is from 2010 to 2023. You can modify the start and end dates in the yfinance download function to focus on different time periods.
Modify Model: You can tweak the LSTM architecture by adjusting the number of LSTM layers, units, or epochs to optimize performance.
Example
Here’s an example of how to run the code:



import yfinance as yf

# Download Apple stock data from Yahoo Finance
df = yf.download('AAPL', start='2010-01-01', end='2023-01-01')
After training the model, the actual and predicted stock prices will be plotted, and the RMSE will be calculated to measure prediction accuracy.

Evaluation
The model’s performance is evaluated using Root Mean Squared Error (RMSE), which gives an indication of how well the model performs in predicting the stock price.

python

from sklearn.metrics import mean_squared_error

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'RMSE: {rmse}')

#Results
The model provides a basic framework for predicting stock prices, though stock market predictions are inherently uncertain due to the market's volatility and complexity. The LSTM model can serve as a foundation for more sophisticated models that may include additional features such as trading volume, news sentiment, etc.

#Future Improvements
Hyperparameter Tuning: Optimize the number of layers, units, and epochs to improve the accuracy.
Incorporate Other Features: Include other stock market indicators like trading volume or technical indicators (e.g., moving averages) to improve predictions.
Add News Sentiment Analysis: Incorporate sentiment analysis from news sources to help improve the stock price prediction.

#License
This project is open source and available under the MIT License.

