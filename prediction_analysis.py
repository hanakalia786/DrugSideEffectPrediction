import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics.pairwise import cosine_similarity


# STEP 1: Set up the computer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "data/onsides/"


# STEP 2: Load datasets

fingerprints = pd.read_csv(
    data_path + "morgan_fingerprints.csv",
    dtype={"ingredient_id": str}
)

ingredients = pd.read_csv(
    data_path + "model_ingredients.csv",
    dtype={"ingredient_id": str}
)


# STEP 3: Load training feature and label order

X_template = pd.read_csv(
    data_path + "X_features.csv",
    index_col=0,
    nrows=0
)

feature_columns = X_template.columns.tolist()

y_template = pd.read_csv(
    data_path + "y_labels.csv",
    index_col=0,
    nrows=0
)

label_ids = [str(label) for label in y_template.columns]


# STEP 4: Clean ingredient IDs

fingerprints["ingredient_id"] = (
    fingerprints["ingredient_id"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
)

ingredients["ingredient_id"] = (
    ingredients["ingredient_id"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
)


# STEP 5: Define the MLP model

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


# STEP 6: Load the trained model

model = SideEffectMLP(
    input_size=len(feature_columns),
    output_size=len(label_ids)
).to(device)

model.load_state_dict(
    torch.load(
        data_path + "side_effect_mlp.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()


# STEP 7: Select medicines

medicine_names = [
    "atorvastatin",
    "ibuprofen",
    "aspirin",
    "paracetamol"
]

ingredient_aliases = {
    "paracetamol": "acetaminophen"
}


# STEP 8: Generate predictions

all_predictions = {}

for medicine_name in medicine_names:

    search_name = ingredient_aliases.get(
        medicine_name,
        medicine_name
    )

    ingredient_match = ingredients[
        ingredients["rxnorm_name"].str.lower() == search_name
    ]

    if ingredient_match.empty:
        print("Ingredient not found:", medicine_name)
        continue

    ingredient_id = ingredient_match.iloc[0]["ingredient_id"]

    fingerprint_match = fingerprints[
        fingerprints["ingredient_id"] == ingredient_id
    ]

    if fingerprint_match.empty:
        print("Fingerprint not found:", medicine_name)
        continue

    fingerprint_values = (
        fingerprint_match.iloc[0][feature_columns]
        .to_numpy(dtype=np.float32)
    )

    input_tensor = torch.tensor(
        fingerprint_values
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(input_tensor)

        scores = torch.sigmoid(output).cpu().numpy()[0]

    all_predictions[medicine_name] = scores


# STEP 9: Compare prediction similarity

prediction_table = pd.DataFrame(
    all_predictions,
    index=label_ids
)

print("\nPREDICTION SCORE SIMILARITY")
print("===========================")

similarity_matrix = cosine_similarity(
    prediction_table.T
)

similarity_results = pd.DataFrame(
    similarity_matrix,
    index=prediction_table.columns,
    columns=prediction_table.columns
)

print("\nCosine similarity between medicines:")
print(similarity_results.round(4))


# STEP 10: Compare average scores

print("\nAVERAGE MODEL SCORES")
print("====================")

print(prediction_table.mean().round(4))


# STEP 11: Compare score variation

print("\nSCORE VARIATION")
print("===============")

print("Standard deviation for each medicine:")

print(prediction_table.std().round(4))


# STEP 12: Count different top predictions

print("\nNUMBER OF DIFFERENT TOP 10 PREDICTIONS")
print("======================================")

for medicine in prediction_table.columns:

    top_10 = prediction_table[medicine].nlargest(10)

    print(medicine, ":", len(top_10), "predictions")


print("\nAnalysis completed!")
