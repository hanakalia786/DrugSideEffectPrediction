import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import precision_score, recall_score, f1_score


# STEP 1: Set up the computer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "data/onsides/"


# STEP 2: Load validation data

X_validation = pd.read_csv(
    data_path + "X_validation.csv",
    index_col=0
)

y_validation = pd.read_csv(
    data_path + "y_validation.csv",
    index_col=0
)

X_tensor = torch.tensor(
    X_validation.values.astype(np.float32)
).to(device)

actual = y_validation.values.astype(int)


# STEP 3: Define the MLP

class SideEffectMLP(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.network(x)


# STEP 4: Load the original trained model

model = SideEffectMLP(
    input_size=X_validation.shape[1],
    output_size=y_validation.shape[1]
).to(device)

model.load_state_dict(
    torch.load(
        data_path + "side_effect_mlp.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()


# STEP 5: Generate model scores

with torch.no_grad():

    outputs = model(X_tensor)

    probabilities = torch.sigmoid(outputs).cpu().numpy()


# STEP 6: Find the best threshold for each side effect

threshold_values = np.arange(0.05, 0.51, 0.01)

best_thresholds = []

print("\nINDIVIDUAL THRESHOLD OPTIMISATION")
print("================================")

for label_index in range(actual.shape[1]):

    true_labels = actual[:, label_index]
    scores = probabilities[:, label_index]

    # Skip labels with no positive examples
    if true_labels.sum() == 0:

        best_thresholds.append(0.2)
        continue

    best_threshold = 0.2
    best_f1 = -1

    for threshold in threshold_values:

        predictions = (scores >= threshold).astype(int)

        current_f1 = f1_score(
            true_labels,
            predictions,
            zero_division=0
        )

        if current_f1 > best_f1:

            best_f1 = current_f1
            best_threshold = threshold

    best_thresholds.append(best_threshold)


# STEP 7: Save the individual thresholds

threshold_data = pd.DataFrame({
    "side_effect_id": y_validation.columns,
    "threshold": best_thresholds
})

threshold_data.to_csv(
    data_path + "side_effect_thresholds.csv",
    index=False
)

print("\nIndividual thresholds saved successfully!")


# STEP 8: Compare individual thresholds against 0.2

individual_predictions = (
    probabilities >= np.array(best_thresholds)
).astype(int)

original_predictions = (
    probabilities >= 0.2
).astype(int)


def calculate_metrics(actual, predictions):

    return {
        "Precision": precision_score(
            actual,
            predictions,
            average="micro",
            zero_division=0
        ),

        "Recall": recall_score(
            actual,
            predictions,
            average="micro",
            zero_division=0
        ),

        "F1 Score": f1_score(
            actual,
            predictions,
            average="micro",
            zero_division=0
        )
    }


original_results = calculate_metrics(
    actual,
    original_predictions
)

individual_results = calculate_metrics(
    actual,
    individual_predictions
)


# STEP 9: Display comparison

print("\nVALIDATION PERFORMANCE COMPARISON")
print("=================================")

print("\nOriginal threshold (0.2):")

for metric, value in original_results.items():
    print(metric, ":", round(value, 4))


print("\nIndividual thresholds:")

for metric, value in individual_results.items():
    print(metric, ":", round(value, 4))


print("\nThreshold optimisation completed!")
