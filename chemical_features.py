
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def generate_morgan_fingerprints():

    chemical_clean = pd.read_csv(
        "data/onsides/chemical_structures_clean.csv",
        dtype={"ingredient_id": str}
    )

    model_ingredients = pd.read_csv(
        "data/onsides/model_ingredients.csv",
        dtype={"ingredient_id": str}
    )

    # Match ingredient IDs
    chemical_clean["ingredient_id"] = (
        chemical_clean["ingredient_id"].str.replace(
            r"\.0$", "", regex=True
        )
    )

    model_ingredients["ingredient_id"] = (
        model_ingredients["ingredient_id"].str.replace(
            r"\.0$", "", regex=True
        )
    )

    # Keep ingredients with available structures
    available = model_ingredients.merge(
        chemical_clean[
            ["ingredient_id", "smiles", "status"]
        ],
        on="ingredient_id",
        how="inner"
    )

    available = available[
        available["status"] == "Retrieved"
    ]

    # Morgan fingerprint generator
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048
    )

    fingerprint_data = []

    for _, row in available.iterrows():

        mol = Chem.MolFromSmiles(row["smiles"])

        if mol is None:
            continue

        fingerprint = generator.GetFingerprint(mol)

        fingerprint_array = np.zeros(2048, dtype=int)

        DataStructs.ConvertToNumpyArray(
            fingerprint,
            fingerprint_array
        )

        fingerprint_data.append(
            [row["ingredient_id"], row["rxnorm_name"]]
            + fingerprint_array.tolist()
        )

    # Create DataFrame
    columns = (
        ["ingredient_id", "rxnorm_name"]
        + [f"feature_{i}" for i in range(2048)]
    )

    fingerprint_df = pd.DataFrame(
        fingerprint_data,
        columns=columns
    )

    # Save results
    fingerprint_df.to_csv(
        "data/onsides/morgan_fingerprints.csv",
        index=False
    )

    print("\nMorgan fingerprints generated!")
    print("Total fingerprints:", len(fingerprint_df))
    print("Fingerprint features: 2048")

    return fingerprint_df
