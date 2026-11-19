import pandas as pd


def prepare_training_data():

    # Load the fingerprint features
    fingerprints = pd.read_csv(
        "data/onsides/morgan_fingerprints.csv",
        dtype={"ingredient_id": str}
    )

    # Load the side-effect data
    side_effects = pd.read_csv(
        "data/onsides/model_training_data.csv",
        dtype={"ingredient_id": str}
    )

    # Clean ingredient IDs
    fingerprints["ingredient_id"] = fingerprints["ingredient_id"].str.replace(
        r"\.0$", "", regex=True
    )

    side_effects["ingredient_id"] = side_effects["ingredient_id"].str.replace(
        r"\.0$", "", regex=True
    )

    # Keep only ingredients that have chemical fingerprints
    matched_ids = sorted(
        set(fingerprints["ingredient_id"]) &
        set(side_effects["ingredient_id"])
    )

    fingerprints = fingerprints[
        fingerprints["ingredient_id"].isin(matched_ids)
    ].drop_duplicates("ingredient_id").set_index("ingredient_id")

    side_effects = side_effects[
        side_effects["ingredient_id"].isin(matched_ids)
    ]

    # Create one column for each side effect
    labels = sorted(side_effects["effect_meddra_id"].unique())

    y = (
        side_effects
        .drop_duplicates(["ingredient_id", "effect_meddra_id"])
        .assign(value=1)
        .pivot(index="ingredient_id",
               columns="effect_meddra_id",
               values="value")
        .reindex(index=matched_ids, columns=labels)
        .fillna(0)
    )

    # Align fingerprint rows with the same ingredient order
    X = fingerprints.reindex(matched_ids).drop(
        columns=["rxnorm_name"], errors="ignore"
    )

    # Save the prepared datasets
    X.to_csv("data/onsides/X_features.csv")
    y.to_csv("data/onsides/y_labels.csv")

    print("\nTraining data prepared!")
    print("Ingredients:", len(matched_ids))
    print("Fingerprint features:", X.shape[1])
    print("Side-effect labels:", y.shape[1])
    print("X shape:", X.shape)
    print("Y shape:", y.shape)


if __name__ == "__main__":
    prepare_training_data()