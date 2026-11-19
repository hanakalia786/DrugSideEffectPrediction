import pandas as pd
# ==========================================
# LOAD THE ORIGINAL DATASETS
# ==========================================

# Loads the original dataset including the side effects

data_path = "data/onsides/product_adverse_effect.csv"

data = pd.read_csv(
    data_path,
    dtype={
        "product_label_id": str,
        "effect_meddra_id": str
    }
)

print("Dataset loaded")
print(data.head())

print("\nDataset shape:", data.shape)


# ==========================================
# UNDERSTANDING THE SIDE-EFFECT NAMES
# ==========================================

vocab_path = "data/onsides/vocab_meddra_adverse_effect.csv"

vocab = pd.read_csv(
    vocab_path,
    dtype={"meddra_id": str}
)

print("\nSide-effect vocabulary:")
print(vocab.head())

print("\nVocabulary columns:")
print(vocab.columns.tolist())


# ==========================================
# KEEPS ONLY THE MAIN SIDE-EFFECT NAMES
# ==========================================

main_effects = vocab[
    vocab["meddra_term_type"] == "PT"
].copy()

print("\nNumber of main side-effect names:", len(main_effects))

print("\nExamples of side effects:")
print(main_effects[["meddra_id", "meddra_name"]].head(10))


# ==========================================
# CONNECTING MEDICINES TO SIDE EFFECTS
# ==========================================

# Loads the medicine mapping file

medicine_path = "data/onsides/product_to_rxnorm.csv"

medicines = pd.read_csv(
    medicine_path,
    dtype=str
)

print("\nMedicine mapping:")
print(medicines.head())


# ==========================================
# CHECK THE INGREDIENT INFORMATION
# ==========================================

# Loads the medicine ingredient information

ingredient_path = "data/onsides/vocab_rxnorm_ingredient_to_product.csv"

ingredients = pd.read_csv(
    ingredient_path,
    dtype=str
)

print("\nIngredient information:")
print(ingredients.head())


# ==========================================
# FIND THE ACTUAL INGREDIENT NAME
# ==========================================

# Translates ingredient IDs into readable names

ingredient_names_path = "data/onsides/vocab_rxnorm_ingredient.csv"

ingredient_names = pd.read_csv(
    ingredient_names_path,
    dtype=str
)

print("\nIngredient names:")
print(ingredient_names.head())


# ==========================================
# PUT THE MEDICINE INFORMATION TOGETHER
# ==========================================

# Connect each medicine product to its ingredient name

product_ingredients = ingredients.merge(
    ingredient_names,
    left_on="ingredient_id",
    right_on="rxnorm_id",
    how="left"
)

print("\nProducts connected to ingredients:")
print(
    product_ingredients[
        ["product_id", "rxnorm_name"]
    ].head(10)
)

print("\nTotal product-ingredient connections:")
print(len(product_ingredients))

print("\nUnique ingredients:")
print(product_ingredients["rxnorm_name"].nunique())


# ==========================================
# CHECK HOW MANY PRODUCTS HAVE INGREDIENTS
# ==========================================

ingredient_counts = ingredients.groupby(
    "product_id"
)["ingredient_id"].nunique()

print("\nProducts with ingredients:", ingredients["product_id"].nunique())

print(
    "Products without ingredients:",
    ingredients["product_id"].nunique()
    - ingredients["product_id"].nunique()
)


# ==========================================
# CONNECT THE SIDE EFFECTS TO THE MEDICINE LABELS
# ==========================================

# Connect the original side-effect records to the medicine mapping

side_effect_data = data.merge(
    medicines,
    left_on="product_label_id",
    right_on="label_id",
    how="inner"
)

print("\nConnected side-effect records:")
print(
    side_effect_data[
        ["product_label_id", "rxnorm_product_id", "effect_meddra_id"]
    ].head(10)
)

print("\nTotal connected records:", len(side_effect_data))


# ==========================================
# CHECK HOW MANY PRODUCTS EACH LABEL CONNECTS TO
# ==========================================

product_counts = medicines.groupby(
    "label_id"
)["rxnorm_product_id"].nunique()

print("\nNumber of labels:", len(product_counts))
print("Labels with multiple products:", (product_counts > 1).sum())
print("Maximum products per label:", product_counts.max())


# ==========================================
# CHECK PRODUCTS WITH MULTIPLE INGREDIENTS
# ==========================================

ingredient_counts = ingredients.groupby(
    "product_id"
)["ingredient_id"].nunique()

print("\nProducts with one ingredient:")
print((ingredient_counts == 1).sum())

print("\nProducts with multiple ingredients:")
print((ingredient_counts > 1).sum())

print("\nMaximum ingredients in one product:")
print(ingredient_counts.max())


# ==========================================
# CHECK INGREDIENT MAPPING COMPLETENESS
# ==========================================

print("\nChecking ingredient mapping completeness...")

# Get all products linked to each original label

label_product_mapping = medicines[
    ["label_id", "rxnorm_product_id"]
].drop_duplicates()

# Get all products that have ingredient mappings

product_ingredient_mapping = ingredients[
    ["product_id", "ingredient_id"]
].drop_duplicates()

# Connect labels to their mapped products

label_product_ingredients = label_product_mapping.merge(
    product_ingredient_mapping,
    left_on="rxnorm_product_id",
    right_on="product_id",
    how="left"
)

# Count all products and products with ingredient mappings per label

mapping_check = label_product_ingredients.groupby("label_id").agg(
    total_products=("rxnorm_product_id", "nunique"),
    mapped_products=("product_id", "nunique"),
    unique_ingredients=("ingredient_id", "nunique")
)

# Keep labels where every product has an ingredient mapping

complete_labels = mapping_check[
    (mapping_check["total_products"] == mapping_check["mapped_products"]) &
    (mapping_check["unique_ingredients"] == 1)
]

print("\nLabels with complete mappings and exactly one ingredient:")
print(len(complete_labels))

print("\nLabels excluded because of incomplete or multiple ingredient mappings:")
print(len(mapping_check) - len(complete_labels))

print("\nMapping completeness check finished!")


# ==========================================
# CREATE VALIDATED TRAINING DATASET
# ==========================================

print("\nCreating validated ingredient-level dataset...")

# Keep the original label and its side-effect relationship

label_effects = data[
    ["product_label_id", "effect_meddra_id"]
].drop_duplicates()

# Get labels with complete mappings and exactly one ingredient

complete_label_ids = complete_labels.index.astype(str)

# Keep original side-effect records for eligible labels

validated_label_effects = label_effects[
    label_effects["product_label_id"].isin(complete_label_ids)
].copy()

# Connect eligible labels to their products

validated_data = validated_label_effects.merge(
    label_product_mapping,
    left_on="product_label_id",
    right_on="label_id",
    how="inner"
)

# Connect products to their ingredients

validated_data = validated_data.merge(
    product_ingredient_mapping,
    left_on="rxnorm_product_id",
    right_on="product_id",
    how="inner"
)

# Add readable ingredient names

validated_data = validated_data.merge(
    ingredient_names[["rxnorm_id", "rxnorm_name"]],
    left_on="ingredient_id",
    right_on="rxnorm_id",
    how="inner"
)

# Add readable side-effect names

validated_data = validated_data.merge(
    main_effects[["meddra_id", "meddra_name"]],
    left_on="effect_meddra_id",
    right_on="meddra_id",
    how="inner"
)

# Keep unique ingredient-side-effect relationships

validated_data = validated_data.drop_duplicates(
    subset=["ingredient_id", "effect_meddra_id"]
)

# Display results

print("\nValidated dataset:")
print(validated_data.head(10))

print("\nDataset shape:")
print(validated_data.shape)

print("\nNumber of original labels:")
print(validated_data["product_label_id"].nunique())

print("\nNumber of unique ingredients:")
print(validated_data["ingredient_id"].nunique())

print("\nNumber of unique side effects:")
print(validated_data["effect_meddra_id"].nunique())

print("\nNumber of ingredient-side-effect pairs:")
print(
    validated_data[
        ["ingredient_id", "effect_meddra_id"]
    ].drop_duplicates().shape[0]
)

# Save validated dataset

validated_data.to_csv(
    "data/onsides/validated_ingredient_side_effects.csv",
    index=False
)

print("\nValidated dataset saved successfully!")


# ==========================================
# EXPLORE THE VALIDATED DATASET
# ==========================================

print("\nExploring side-effect distribution...")

# Count how many ingredients are associated with each side effect

side_effect_frequency = (
    validated_data.groupby("meddra_name")["ingredient_id"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nTop 20 most common side effects:")
print(side_effect_frequency.head(20))


# Count the number of distinct side effects for each ingredient

ingredient_frequency = (
    validated_data.groupby("rxnorm_name")["effect_meddra_id"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nTop 20 ingredients with the most side effects:")
print(ingredient_frequency.head(20))


# Check the overall distribution

print("\nSide-effect distribution summary:")
print(side_effect_frequency.describe())

print("\nTotal unique side effects:")
print(len(side_effect_frequency))

print("\nTotal unique ingredients:")
print(len(ingredient_frequency))

print("\nDataset exploration completed!")


# ==========================================
# CHECK SIDE-EFFECT FREQUENCY THRESHOLDS
# ==========================================

print("\nChecking side-effect frequency thresholds...")

thresholds = [5, 10, 20, 30, 50]

for threshold in thresholds:

    selected_effects = side_effect_frequency[
        side_effect_frequency >= threshold
    ]

    print(f"\nMinimum ingredients: {threshold}")
    print("Side effects retained:", len(selected_effects))
    print("Side effects removed:", len(side_effect_frequency) - len(selected_effects))

print("\nThreshold investigation completed!")


# ==========================================
# SELECT SIDE EFFECTS FOR MODEL TRAINING
# ==========================================

print("\nPreparing side effects for model training...")

# Count distinct ingredients associated with each side-effect ID

effect_frequency = (
    validated_data.groupby("effect_meddra_id")["ingredient_id"]
    .nunique()
    .sort_values(ascending=False)
)

# Set minimum frequency threshold

minimum_frequency = 20

# Select side effects meeting the threshold

selected_effect_ids = effect_frequency[
    effect_frequency >= minimum_frequency
].index

# Filter the validated dataset

model_data = validated_data[
    validated_data["effect_meddra_id"].isin(selected_effect_ids)
].copy()

# Remove duplicate ingredient-side-effect pairs

model_data = model_data.drop_duplicates(
    subset=["ingredient_id", "effect_meddra_id"]
)

print("\nFiltered model dataset:")
print(model_data.head(10))

print("\nDataset shape:")
print(model_data.shape)

print("\nUnique ingredients:")
print(model_data["ingredient_id"].nunique())

print("\nSelected side effects:")
print(model_data["effect_meddra_id"].nunique())

print("\nMinimum frequency:")
print(minimum_frequency)

# Save filtered dataset

model_data.to_csv(
    "data/onsides/model_training_data.csv",
    index=False
)

print("\nModel training dataset saved successfully!")


# ==========================================
# PREPARE INGREDIENTS FOR CHEMICAL FEATURES
# ==========================================

print("\nPreparing ingredient list...")

# Get each unique ingredient and its readable name

model_ingredients = model_data[
    ["ingredient_id", "rxnorm_name"]
].drop_duplicates()

# Remove any missing ingredient names

model_ingredients = model_ingredients.dropna(
    subset=["rxnorm_name"]
)

# Display results

print("\nIngredient list:")
print(model_ingredients.head(20))

print("\nTotal unique ingredients:")
print(len(model_ingredients))

print("\nMissing ingredient names:")
print(model_ingredients["rxnorm_name"].isna().sum())

# Save ingredient list

model_ingredients.to_csv(
    "data/onsides/model_ingredients.csv",
    index=False
)

print("\nIngredient list saved successfully!")

print("\nData preparation completed successfully!")
