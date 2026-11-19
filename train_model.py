import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader


# STEP 1: Choose CPU or GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "data/onsides/"


# STEP 2: Load training and validation data

X_train = pd.read_csv(data_path + "X_train.csv", index_col=0)
y_train = pd.read_csv(data_path + "y_train.csv", index_col=0)

X_validation = pd.read_csv(
    data_path + "X_validation.csv", index_col=0
)

y_validation = pd.read_csv(
    data_path + "y_validation.csv", index_col=0
)


# STEP 3: Convert data into tensors

X_train = torch.tensor(
    X_train.values.astype(np.float32)
)

y_train = torch.tensor(
    y_train.values.astype(np.float32)
)

X_validation = torch.tensor(
    X_validation.values.astype(np.float32)
).to(device)

y_validation = torch.tensor(
    y_validation.values.astype(np.float32)
).to(device)


# STEP 4: Prepare training batches

dataset = TensorDataset(X_train, y_train)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


# STEP 5: Set up the MLP model

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


model = SideEffectMLP(
    X_train.shape[1],
    y_train.shape[1]
).to(device)


# STEP 6: Give more importance to rare side effects

# Count how often each side effect appears
side_effect_counts = y_train.sum(dim=0)

# Count how often each side effect does not appear
negative_counts = len(y_train) - side_effect_counts

# Calculate weights for each side effect
weights = negative_counts / side_effect_counts.clamp(min=1)

# Prevent weights from becoming too large
weights = torch.clamp(weights, max=10)

weights = weights.to(device)


# STEP 7: Set up loss function and optimiser

criterion = nn.BCEWithLogitsLoss(
    pos_weight=weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# STEP 8: Train the model

epochs = 30
patience = 5

best_validation_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for batch_X, batch_y in loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    training_loss = total_loss / len(loader)


    # Check how well the model performs on validation data

    model.eval()

    with torch.no_grad():

        validation_outputs = model(X_validation)

        validation_loss = criterion(
            validation_outputs,
            y_validation
        )

    validation_loss = validation_loss.item()

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Training Loss: {training_loss:.4f}, "
        f"Validation Loss: {validation_loss:.4f}"
    )


    # Save the best version of the model

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        torch.save(
            model.state_dict(),
            data_path + "side_effect_mlp_weighted_v2.pth"
        )

        epochs_without_improvement = 0

        print("Best model saved!")

    else:

        epochs_without_improvement += 1


    # Stop training if validation loss stops improving

    if epochs_without_improvement >= patience:

        print("\nEarly stopping triggered!")
        break


# STEP 9: Finish

print("\nWeighted model training completed!")

print(
    "Best validation loss:",
    round(best_validation_loss, 4)
)

print("Model saved as side_effect_mlp_weighted_v2.pth")
