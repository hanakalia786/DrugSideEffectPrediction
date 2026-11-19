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

# Choose CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "data/onsides/"

# STEP 1: Load validation data

X_validation = pd.read_csv(
    data_path + "X_validation.csv", index_col=0
)

y_validation = pd.read_csv(
    data_path + "y_validation.csv", index_col=0
)

# Convert data into tensors

X_tensor = torch.tensor(
    X_validation.values.astype(np.float32)
).to(device)

real_answers = y_validation.values.astype(int)


# STEP 2: Set up our MLP model

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


# STEP 3: Create a function to test each model

def test_model(model_file, model_name):

    model = SideEffectMLP(
        X_validation.shape[1],
        y_validation.shape[1]
    ).to(device)

    model.load_state_dict(
        torch.load(
            data_path + model_file,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    # Ask the model to make predictions
    with torch.no_grad():
        results = model(X_tensor)
        probabilities = torch.sigmoid(results).cpu().numpy()

    # Use the same threshold for both models
    predictions = (probabilities >= 0.2).astype(int)

    # Calculate performance
    precision = precision_score(
        real_answers, predictions,
        average="micro", zero_division=0
    )

    recall = recall_score(
        real_answers, predictions,
        average="micro", zero_division=0
    )

    f1 = f1_score(
        real_answers, predictions,
        average="micro", zero_division=0
    )

    macro_f1 = f1_score(
        real_answers, predictions,
        average="macro", zero_division=0
    )

    loss = hamming_loss(real_answers, predictions)

    print("\n" + model_name)
    print("----------------------")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Hamming Loss:", round(loss, 4))


# STEP 4: Compare both models

test_model(
    "side_effect_mlp.pth",
    "ORIGINAL MLP"
)

test_model(
    "side_effect_mlp_weighted.pth",
    "WEIGHTED MLP"
)

test_model(
    "side_effect_mlp_weighted_v2.pth",
    "CORRECTED WEIGHTED MLP"
)

print("\nModel comparison completed!")
