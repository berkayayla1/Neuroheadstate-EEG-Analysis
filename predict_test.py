import pandas as pd
import joblib

# 1. Load the Trained Model
print("Loading the trained model...")
model = joblib.load('model.pkl')

# 2. Load the Test Data
print("Loading test.csv data...")
try:
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("ERROR: 'test.csv' file not found! Please put it in the EYEPROJECT folder.")
    exit()

# 3. Prepare Data
# The model expects only sensor data. 
# If there is an 'id' column (common in Kaggle), we must remove it before prediction.
feature_columns = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

# Select only the columns that the model knows
try:
    X_test = test_df[feature_columns]
    print("Data prepared successfully.")
except KeyError:
    print("ERROR: Column names in test.csv do not match the training data!")
    print("Expected columns:", feature_columns)
    exit()

# 4. Make Predictions
print("Making predictions... (This might take a moment)")
predictions = model.predict(X_test)

# 5. Save Results
# We create a new table with the inputs and the prediction result
results_df = test_df.copy()
results_df['Predicted_Eye_State'] = predictions # 0 = Open, 1 = Closed

# Save to a new CSV file
results_df.to_csv('test_results.csv', index=False)

print("SUCCESS! Predictions saved to 'test_results.csv'.")
print("You can open this file in Excel to see the results.")
