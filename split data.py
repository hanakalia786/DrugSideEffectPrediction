import pandas as pd
from sklearn.model_selection import train_test_split

data_path = "data/onsides/"

# Load features and labels
X = pd.read_csv(data_path + "X_features.csv", index_col=0)
y = pd.read_csv(data_path + "y_labels.csv", index_col=0)

# Make ingredient IDs consistent
X.index = X.index.astype(str).str.strip()
y.index = y.index.astype(str).str.strip()

# Keep only matching ingredients
common_ids = X.index.intersection(y.index)

X = X.loc[common_ids]
y = y.loc[common_ids]

# First split: 80% development data, 20% test data
X_development, X_test, y_development, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Second split: divide development data into 70% training
# and 10% validation of the original dataset
X_train, X_validation, y_train, y_validation = train_test_split(
    X_development,
    y_development,
    test_size=0.125,
    random_state=42
)

# Save all datasets
X_train.to_csv(data_path + "X_train.csv")
X_validation.to_csv(data_path + "X_validation.csv")
X_test.to_csv(data_path + "X_test.csv")

y_train.to_csv(data_path + "y_train.csv")
y_validation.to_csv(data_path + "y_validation.csv")
y_test.to_csv(data_path + "y_test.csv")

print("Data split completed!")
print("Training ingredients:", len(X_train))
print("Validation ingredients:", len(X_validation))
print("Testing ingredients:", len(X_test))

print("X_train:", X_train.shape)
print("X_validation:", X_validation.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_validation:", y_validation.shape)
print("y_test:", y_test.shape)

