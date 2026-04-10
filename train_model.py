import pandas as pd
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv('train.csv')

# 2. Prepare Data (X: Sensors, y: Eye State)
X = df.drop('eyeDetection', axis=1)
y = df['eyeDetection']

# 3. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Best Model (LabelSpreading)
# We found this was the best model using LazyPredict Analysis
print("Training LabelSpreading model... (This might take a few seconds)")
model = LabelSpreading(kernel='knn', n_neighbors=7)
model.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the Model
joblib.dump(model, 'model.pkl')
print("Success! Model saved as 'model.pkl'")
