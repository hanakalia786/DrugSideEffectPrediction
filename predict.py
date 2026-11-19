import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import tkinter as tk
from PIL import ImageTk
from rdkit import Chem
from rdkit.Chem import Draw

# STEP 1: Set up the computer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "data/onsides/"


# STEP 2: Load the datasets

fingerprints = pd.read_csv(
    data_path + "morgan_fingerprints.csv",
    dtype={"ingredient_id": str}
)

ingredients = pd.read_csv(
    data_path + "model_ingredients.csv",
    dtype={"ingredient_id": str}
)

side_effects = pd.read_csv(
    data_path + "model_training_data.csv",
    dtype={"ingredient_id": str}
)

chemical_data = pd.read_csv(
    data_path + "chemical_structures_clean.csv",
    dtype={"ingredient_id": str}
)

chemical_data["ingredient_id"] = (
    chemical_data["ingredient_id"]
    .str.replace(r"\.0$", "", regex=True)
)
# STEP 3: Load the training feature order

X_template = pd.read_csv(
    data_path + "X_features.csv",
    index_col=0,
    nrows=0
)

feature_columns = X_template.columns.tolist()


# STEP 4: Load the side-effect label order

y_template = pd.read_csv(
    data_path + "y_labels.csv",
    index_col=0,
    nrows=0
)

label_ids = [str(label) for label in y_template.columns]

side_effects["effect_meddra_id"] = (
    side_effects["effect_meddra_id"].astype(str)
)

side_effect_lookup = (
    side_effects[
        ["effect_meddra_id", "meddra_name"]
    ]
    .drop_duplicates(subset="effect_meddra_id")
    .set_index("effect_meddra_id")["meddra_name"]
)

missing_labels = [
    label for label in label_ids
    if label not in side_effect_lookup.index
]

if missing_labels:
    raise ValueError(
        "Some side-effect IDs have no matching names."
    )

side_effect_names = [
    side_effect_lookup.loc[label]
    for label in label_ids
]


# STEP 5: Clean ingredient IDs

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


# STEP 6: Define the MLP model

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


# STEP 7: Load the trained model

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

root = tk.Tk()
root.withdraw()


def show_medicine_popup(medicine_name, formula, smiles, predictions):
    window = tk.Toplevel(root)
    window.title(f"{medicine_name.title()} - AI Side Effect Prediction")
    window.geometry("650x750")

    # Medicine name
    tk.Label(
        window,
        text=medicine_name.title(),
        font=("Arial", 20, "bold")
    ).pack(pady=10)

    # Molecular formula
    tk.Label(
        window,
        text=f"Molecular Formula: {formula}",
        font=("Arial", 12)
    ).pack(pady=5)

    # Chemical structure
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is not None:
        image = Draw.MolToImage(molecule, size=(400, 250))
        photo = ImageTk.PhotoImage(image)

        structure_label = tk.Label(window, image=photo)
        structure_label.image = photo
        structure_label.pack(pady=10)
    else:
        tk.Label(
            window,
            text="Chemical structure could not be displayed."
        ).pack()

    # Predicted side effects
    tk.Label(
        window,
        text="Top 10 Predicted Side Effects",
        font=("Arial", 15, "bold")
    ).pack(pady=10)

    for effect, score in predictions:
        tk.Label(
            window,
            text=f"{effect} - {score:.2f}% model score",
            font=("Arial", 11)
        ).pack(anchor="w", padx=80)

    # Disclaimer
    tk.Label(
        window,
        text="Model predictions are not confirmed clinical risks.",
        font=("Arial", 9, "italic")
    ).pack(pady=15)

    # Close using the X in the top-right corner
    window.protocol("WM_DELETE_WINDOW", window.destroy)

    # Return to console after closing
    root.wait_window(window)


# STEP 8: Create alternative ingredient names

ingredient_aliases = {
    "paracetamol": "acetaminophen"
}


# STEP 9: Keep asking for ingredients

print("\n===================================")
print("   AI SIDE EFFECT PREDICTION SYSTEM")
print("===================================")
print("Enter a medicine ingredient to begin.")
print("Type 'exit' to close the program.")


while True:

    medicine_name = input(
        "\nEnter a medicine ingredient name: "
    ).strip().lower()

    if medicine_name == "exit":
        print("\nThank you for using the prediction system!")
        break

    if not medicine_name:
        print("Please enter an ingredient name.")
        continue

    # Recognise alternative ingredient names
    search_name = ingredient_aliases.get(
        medicine_name,
        medicine_name
    )

    # Find the ingredient
    ingredient_match = ingredients[
        ingredients["rxnorm_name"].str.lower() == search_name
    ]

    if ingredient_match.empty:
        print("\nSorry, this ingredient is not in our dataset.")
        continue

    ingredient_id = ingredient_match.iloc[0]["ingredient_id"]

    # Find its chemical fingerprint
    fingerprint_match = fingerprints[
        fingerprints["ingredient_id"] == ingredient_id
    ]

    if fingerprint_match.empty:
        print("\nNo chemical fingerprint found for this ingredient.")
        continue

    # Check fingerprint columns
    missing_features = [
        column for column in feature_columns
        if column not in fingerprints.columns
    ]

    if missing_features:
        raise ValueError(
            "Some training fingerprint columns are missing."
        )

    # Get fingerprint values
    fingerprint_values = (
        fingerprint_match.iloc[0][feature_columns]
        .to_numpy(dtype=np.float32)
    )

    input_tensor = torch.tensor(
        fingerprint_values
    ).unsqueeze(0).to(device)

    # Predict side effects
    with torch.no_grad():
        output = model(input_tensor)
        model_scores = torch.sigmoid(output).cpu().numpy()[0]

    # Get the 10 highest predictions
    top_predictions = np.argsort(model_scores)[::-1][:10]

    # Display results in console
    print("\nPredicted side effects for:", medicine_name)
    print("--------------------------------")

    for index in top_predictions:
        print(
            side_effect_names[index],
            "-",
            round(float(model_scores[index]) * 100, 2),
            "% model score"
        )

    # Prepare predictions for the pop-up
    predictions = [
        (side_effect_names[index], float(model_scores[index]) * 100)
        for index in top_predictions
    ]

    # Find chemical information
    chemical_match = chemical_data[
        chemical_data["ingredient_id"] == ingredient_id
    ]

    if not chemical_match.empty:
        chemical_info = chemical_match.iloc[0]

        formula = chemical_info["molecular_formula"]
        smiles = chemical_info["smiles"]

        if pd.notna(formula) and pd.notna(smiles):
            show_medicine_popup(
                medicine_name,
                formula,
                smiles,
                predictions
            )
        else:
            print("Chemical formula or structure is unavailable.")
    else:
        print("Chemical information is unavailable.")

    print("\nPrediction completed!")
    print("These are model scores, not confirmed clinical risks.")


# Close the hidden Tkinter window when the program exits
root.destroy()

