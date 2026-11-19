import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# STEP 1: Load datasets

data_path = "data/onsides/"

fingerprints = pd.read_csv(
    data_path + "morgan_fingerprints.csv",
    dtype={"ingredient_id": str}
)

ingredients = pd.read_csv(
    data_path + "model_ingredients.csv",
    dtype={"ingredient_id": str}
)


# STEP 2: Load the exact fingerprint columns used in training

X_template = pd.read_csv(
    data_path + "X_features.csv",
    index_col=0,
    nrows=0
)

fingerprint_columns = X_template.columns.tolist()

print("Number of fingerprint features:", len(fingerprint_columns))


# STEP 3: Clean ingredient IDs

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


# STEP 4: Select medicines

medicine_names = [
    "atorvastatin",
    "ibuprofen",
    "aspirin",
    "paracetamol"
]

ingredient_aliases = {
    "paracetamol": "acetaminophen"
}


# STEP 5: Extract fingerprints

selected_fingerprints = {}

for medicine in medicine_names:

    search_name = ingredient_aliases.get(medicine, medicine)

    ingredient_match = ingredients[
        ingredients["rxnorm_name"].str.lower() == search_name
    ]

    if ingredient_match.empty:
        print("Ingredient not found:", medicine)
        continue

    ingredient_id = ingredient_match.iloc[0]["ingredient_id"]

    fingerprint_match = fingerprints[
        fingerprints["ingredient_id"] == ingredient_id
    ]

    if fingerprint_match.empty:
        print("Fingerprint not found:", medicine)
        continue

    missing_columns = [
        column for column in fingerprint_columns
        if column not in fingerprints.columns
    ]

    if missing_columns:
        raise ValueError("Fingerprint columns are missing!")

    values = (
        fingerprint_match.iloc[0][fingerprint_columns]
        .to_numpy(dtype=np.float32)
    )

    selected_fingerprints[medicine] = values


# STEP 6: Create fingerprint table

fingerprint_table = pd.DataFrame(
    selected_fingerprints
)


# STEP 7: Count active fingerprint bits

print("\nACTIVE FINGERPRINT BITS")
print("=======================")

for medicine in fingerprint_table.columns:

    active_bits = int(fingerprint_table[medicine].sum())

    print(medicine, ":", active_bits)


# STEP 8: Calculate fingerprint similarity

similarity_matrix = cosine_similarity(
    fingerprint_table.T
)

similarity_results = pd.DataFrame(
    similarity_matrix,
    index=fingerprint_table.columns,
    columns=fingerprint_table.columns
)

print("\nFINGERPRINT SIMILARITY")
print("======================")

print(similarity_results.round(4))


# STEP 9: Count fingerprint differences

print("\nFINGERPRINT DIFFERENCES")
print("=======================")

for i, medicine_a in enumerate(fingerprint_table.columns):

    for medicine_b in fingerprint_table.columns[i + 1:]:

        differences = np.sum(
            fingerprint_table[medicine_a].values !=
            fingerprint_table[medicine_b].values
        )

        print(
            medicine_a,
            "vs",
            medicine_b,
            ":",
            differences,
            "different bits"
        )


print("\nFingerprint analysis completed!")
