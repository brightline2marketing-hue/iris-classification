from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the classic Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a simple model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy:.2f}")

# Show example predictions
print("\nExample predictions on test set:")
for i in range(5):
    sample = X_test[i]
    actual = y_test[i]
    predicted = predictions[i]
    print(
        f"Sample: {sample}, Actual: {iris.target_names[actual]}, "
        f"Predicted: {iris.target_names[predicted]}"
    )

# Predict on new data
new_sample = [[5.1, 3.5, 1.4, 0.2]]
new_prediction = model.predict(new_sample)
print(f"\nNew sample prediction: {iris.target_names[new_prediction[0]]}")
