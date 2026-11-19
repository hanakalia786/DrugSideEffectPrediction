
import pandas as pd
import matplotlib.pyplot as plt


# STEP 1: Load the training dataset

data_path = "data/onsides/"

data = pd.read_csv(
    data_path + "model_training_data.csv"
)


# STEP 2: Count side effects per ingredient

ingredient_effects = (
    data.groupby("ingredient_id")["effect_meddra_id"]
    .nunique()
)

print("\nSIDE EFFECTS PER INGREDIENT")
print("===========================")

print("Number of ingredients:", len(ingredient_effects))
print("Average side effects:", round(ingredient_effects.mean(), 2))
print("Maximum side effects:", ingredient_effects.max())
print("Minimum side effects:", ingredient_effects.min())


# STEP 3: Find the most frequent side effects

effect_counts = (
    data.groupby("meddra_name")["ingredient_id"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nTOP 15 MOST FREQUENT SIDE EFFECTS")
print("=================================")

print(effect_counts.head(15))


# STEP 4: Find the least frequent side effects

print("\nBOTTOM 15 LEAST FREQUENT SIDE EFFECTS")
print("=====================================")

print(effect_counts.tail(15))


# STEP 5: Display the side-effect distribution

plt.figure(figsize=(10, 6))

plt.hist(
    ingredient_effects,
    bins=30
)

plt.title("Distribution of Side Effects per Ingredient")
plt.xlabel("Number of Side Effects")
plt.ylabel("Number of Ingredients")

plt.tight_layout()
plt.show()


# STEP 6: Display the most frequent side effects

plt.figure(figsize=(10, 7))

effect_counts.head(15).sort_values().plot(
    kind="barh"
)

plt.title("15 Most Frequent Side Effects")
plt.xlabel("Number of Ingredients")
plt.ylabel("Side Effect")

plt.tight_layout()
plt.show()


print("\nTraining data analysis completed!")
