
import pandas as pd

data_path = "data/onsides/"

# STEP 1: Load the training features and labels
X_train = pd.read_csv(
    data_path + "X_train.csv",
    index_col=0
)

y_train = pd.read_csv(
    data_path + "y_train.csv",
    index_col=0
)

# STEP 2: Load the fingerprint file
fingerprints = pd.read_csv(
    data_path + "morgan_fingerprints.csv"
)

# STEP 3: Load the side-effect dataset
model_data = pd.read_csv(
    data_path + "model_training_data.csv"
)

# STEP 4: Check the number of features
print("\nFEATURE CHECK")
print("----------------------")

print("Training features:", X_train.shape[1])
print("Fingerprint columns:", fingerprints.shape[1] - 2)

# STEP 5: Check ingredient IDs
print("\nINGREDIENT CHECK")
print("----------------------")

print("Training ingredients:", len(X_train))
print("Fingerprint ingredients:", fingerprints["ingredient_id"].nunique())

# STEP 6: Check side-effect labels
print("\nSIDE-EFFECT CHECK")
print("----------------------")

print("Training labels:", y_train.shape[1])

print(
    "Unique side effects in dataset:",
    model_data["effect_meddra_id"].nunique()
)

# STEP 7: Check for missing values
print("\nMISSING DATA CHECK")
print("----------------------")

print("Missing training features:", X_train.isnull().sum().sum())
print("Missing training labels:", y_train.isnull().sum().sum())

# STEP 8: Check whether training and fingerprint IDs match
training_ids = set(X_train.index.astype(str))

fingerprint_ids = set(
    fingerprints["ingredient_id"].astype(str)
)

print("\nID MATCHING")
print("----------------------")

print("Training IDs found in fingerprints:",
      len(training_ids.intersection(fingerprint_ids)))

print("Training IDs missing from fingerprints:",
      len(training_ids - fingerprint_ids))

print("\nAlignment checks completed!")
