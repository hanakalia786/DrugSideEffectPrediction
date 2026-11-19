
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Chooses whether to use the computer's CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Where data files are stored
data_path = "data/onsides/"


# STEP 1: Load our training and validation data

X_train = pd.read_csv(data_path + "X_train.csv", index_col=0)
y_train = pd.read_csv(data_path + "y_train.csv", index_col=0)

X_validation = pd.read_csv(
    data_path + "X_validation.csv", index_col=0
)
y_validation = pd.read_csv(
    data_path + "y_validation.csv", index_col=0
)


# STEP 2: Set up the same MLP model we trained earlier

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


# STEP 3: Load our saved model

model = SideEffectMLP(
    X_validation.shape[1],
    y_validation.shape[1]
).to(device)

model.load_state_dict(
    torch.load(
        data_path + "side_effect_mlp.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()


# STEP 4: Ask the model to make predictions

X_tensor = torch.tensor(
    X_validation.values.astype(np.float32)
).to(device)

with torch.no_grad():
    results = model(X_tensor)

    probabilities = torch.sigmoid(results).cpu().numpy()


# STEP 5: Compare predictions with the real answers

real_answers = y_validation.values.astype(int)

predictions = (probabilities >= 0.2).astype(int)


# STEP 6: Check how common each side effect is

side_effect_counts = y_train.sum(axis=0)

print("\nSIDE EFFECT FREQUENCY")
print("----------------------")

print("Total side effects:", len(side_effect_counts))

print("Side effects appearing fewer than 5 times:",
      (side_effect_counts < 5).sum())

print("Side effects appearing fewer than 10 times:",
      (side_effect_counts < 10).sum())

print("Side effects appearing fewer than 20 times:",
      (side_effect_counts < 20).sum())


# Show the 10 most common side effects

print("\nMOST COMMON SIDE EFFECTS")

print(side_effect_counts.sort_values(ascending=False).head(10))


# Show the 10 least common side effects

print("\nLEAST COMMON SIDE EFFECTS")

print(side_effect_counts.sort_values().head(10))


# STEP 7: Check how many predictions were correct or incorrect

correct_predictions = (
    (real_answers == 1) & (predictions == 1)
).sum()

wrong_predictions = (
    (real_answers == 0) & (predictions == 1)
).sum()

missed_side_effects = (
    (real_answers == 1) & (predictions == 0)
).sum()


print("\nPREDICTION RESULTS")
print("----------------------")

print("Correct predictions:", correct_predictions)

print("Incorrect predictions:", wrong_predictions)

print("Missed side effects:", missed_side_effects)

print("Actual side effects:", real_answers.sum())

print("Predicted side effects:", predictions.sum())


# STEP 8: Compare the average number of side effects

print("\nAVERAGE SIDE EFFECTS PER INGREDIENT")

print(
    "Actual:",
    round(real_answers.sum(axis=1).mean(), 2)
)

print(
    "Predicted:",
    round(predictions.sum(axis=1).mean(), 2)
)


print("\nDiagnosis completed!")
