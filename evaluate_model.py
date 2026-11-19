import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    hamming_loss
)


# STEP 1: Set up the computer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "data/onsides/"


# STEP 2: Load training and testing data

X_train = pd.read_csv(
    data_path + "X_train.csv",
    index_col=0
)

y_train = pd.read_csv(
    data_path + "y_train.csv",
    index_col=0
)

X_test = pd.read_csv(
    data_path + "X_test.csv",
    index_col=0
)

y_test = pd.read_csv(
    data_path + "y_test.csv",
    index_col=0
)

X_test_tensor = torch.tensor(
    X_test.values.astype(np.float32)
).to(device)

actual = y_test.values.astype(int)


# STEP 3: Define the MLP architecture

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
    input_size=X_test.shape[1],
    output_size=y_test.shape[1]
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

    outputs = model(X_test_tensor)

    probabilities = torch.sigmoid(outputs).cpu().numpy()


# STEP 6: Create evaluation function

def evaluate_model_results(actual, predictions):

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

        "Micro F1": f1_score(
            actual,
            predictions,
            average="micro",
            zero_division=0
        ),

        "Macro F1": f1_score(
            actual,
            predictions,
            average="macro",
            zero_division=0
        ),

        "Hamming Loss": hamming_loss(
            actual,
            predictions
        )
    }


# STEP 7: Evaluate original MLP at threshold 0.2

original_predictions = (
    probabilities >= 0.2
).astype(int)

original_results = evaluate_model_results(
    actual,
    original_predictions
)


# STEP 8: Load individual thresholds

threshold_data = pd.read_csv(
    data_path + "side_effect_thresholds.csv",
    dtype={"side_effect_id": str}
)

threshold_data["side_effect_id"] = (
    threshold_data["side_effect_id"]
    .str.replace(r"\.0$", "", regex=True)
)

test_label_ids = [
    str(label).replace(".0", "")
    for label in y_test.columns
]

threshold_lookup = threshold_data.set_index(
    "side_effect_id"
)["threshold"]

missing_thresholds = [
    label for label in test_label_ids
    if label not in threshold_lookup.index
]

if missing_thresholds:
    raise ValueError(
        "Some test labels do not have matching thresholds."
    )

individual_thresholds = np.array([
    threshold_lookup.loc[label]
    for label in test_label_ids
])

individual_predictions = (
    probabilities >= individual_thresholds
).astype(int)

individual_results = evaluate_model_results(
    actual,
    individual_predictions
)


# STEP 9: Create most-frequent-label baseline

label_frequencies = y_train.sum(axis=0)

average_labels = max(
    1,
    int(round(y_train.sum(axis=1).mean()))
)

top_labels = label_frequencies.nlargest(
    average_labels
).index

baseline_predictions = np.zeros_like(actual)

for label in top_labels:

    label_position = y_test.columns.get_loc(label)

    baseline_predictions[:, label_position] = 1


baseline_results = evaluate_model_results(
    actual,
    baseline_predictions
)


# STEP 10: Display final comparison

comparison = pd.DataFrame({
    "Original MLP (0.2)": original_results,
    "Individual Thresholds": individual_results,
    "Baseline": baseline_results
})

print("\nFINAL TEST PERFORMANCE COMPARISON")
print("=================================")

print(comparison.round(4).to_string())


# STEP 11: Display best result

print("\nEVALUATION COMPLETED!")
print("Original model has been preserved.")
print("All results use the same untouched test dataset.")
