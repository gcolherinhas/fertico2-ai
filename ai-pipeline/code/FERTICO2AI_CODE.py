# ==========================================================
# FERTICO2-AI — Molecular Dynamics + Artificial Intelligence
# ==========================================================
# Institution:
#   Federal University of Goiás (UFG)
#   Institute of Physics
#
# Project Description:
#   Open-source computational platform for the classification
#   and prediction of CO2 capture performance in amino acid
#   ionic liquids (AAILs), integrating Molecular Dynamics (MD)
#   simulations and Machine Learning models.
#
# Version:        v1.0.4
# Release date:   December 21, 2025
# Status:         Stable research release
#
# License:        MIT License
# Repository:     https://github.com/gcolherinhas/fertico2-ai
#
# ==========================================================
# Principal Investigators:
#   Prof. Dr. Guilherme Colherinhas de Oliveira
#   Prof. Dr. Wesley Bueno Cardoso
#   Prof. Dr. Tertius Lima da Fonseca
#   Prof. Dr. Leonardo Bruno Assis Oliveira
#   M.Sc. Karinna Mendanha Soares
#   M.Sc. Lucas de Sousa Silva
#   M.Sc. Henrique de Araujo Chagas
#
# ==========================================================
# This code is part of the FERTICO2-AI project and is
# distributed under the MIT License. It may be freely used,
# modified, and redistributed, provided that proper credit
# is given to the original authors.
# ==========================================================

"""# **Part 01 - Classification of Ionic Liquids for C02 Absorpiton**"""

# ==========================================================
# STAGE 1 — Physically ordered K-Means classification
# of Ionic Liquid systems for CO2 absorption
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------------------------------------
# 1. Load the Excel file for classification: CO2-Class.xlsx
# ----------------------------------------------------------

file_path = "CO2-Class.xlsx"
df = pd.read_excel(file_path)

print("Data successfully loaded.")
print(df.head())

# ----------------------------------------------------------
# 2. Select descriptors for clustering
# ----------------------------------------------------------

features = [
    "Coul-LI-CO2",      # Average Coulomb interaction energy between the ionic liquid (AAIL) and CO₂
    "LJ-LI-CO2",        # Average Lennard-Jones interaction energy between the ionic liquid (AAIL) and CO₂
    "HB-LI-CO2",        # Average number of hydrogen bonds formed between the AAIL and CO₂
    "Lifetime-LI-CO2",  # Mean lifetime of AAIL–CO₂ hydrogen bonds (computed via Luzar–Chandler autocorrelation)
    "DG-LI-CO2",        # Effective free energy of interaction between AAIL and CO₂ (thermodynamic descriptor)
    "Dens",             # Average density of the ionic liquid system (obtained from NPT simulations)
    "TC",               # Fraction or relative count of carbon atoms in the AAIL composition
    "TN",               # Fraction or relative count of nitrogen atoms in the AAIL composition
    "TO",               # Fraction or relative count of oxygen atoms in the AAIL composition
    "TH",               # Fraction or relative count of hydrogen atoms in the AAIL composition
    "MSD",              # Diffusion coefficient of the ionic liquid derived from the mean squared displacement (Einstein relation)
    "Coul",             # Average Coulomb interaction energy among ionic liquid constituents (IL–IL)
    "LJ",               # Average Lennard-Jones interaction energy among ionic liquid constituents (IL–IL)
    "HBs",              # Average total number of internal hydrogen bonds within the ionic liquid
    "DHB",              # Effective hydrogen bond dissociation energy (derived from autocorrelation analysis)
    "LFTHB",            # Mean lifetime of hydrogen bonds in the ionic liquid (Luzar–Chandler method, gmx hbond -ac)
    "C1",               # Molar proportion of the first AAIL component (or first ionic liquid in a mixture)
    "C2"                # Molar proportion of the second AAIL component (or second ionic liquid in a mixture)
]

X = df[features].values

# ----------------------------------------------------------
# 3. Feature scaling
# ----------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "Stage1_StandardScaler.pkl")

# ----------------------------------------------------------
# 4. Elbow Method (k = 2 to 10)
# ----------------------------------------------------------

inertia = []
k_values = np.arange(2, 10)

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# ----------------------------------------------------------
# 5. Plot Elbow curve
# ----------------------------------------------------------

plt.figure(figsize=(7, 5))
plt.plot(k_values, inertia, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()

# ----------------------------------------------------------
# 6. Automatic elbow detection
# ----------------------------------------------------------

inertia = np.array(inertia)
second_derivative = np.diff(inertia, n=2)

optimal_k = k_values[np.argmax(second_derivative) + 1]

print(f"Optimal number of clusters (Elbow Method): k = {optimal_k}")

# ----------------------------------------------------------
# 7. Final K-Means with optimal k
# ----------------------------------------------------------

kmeans_final = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=50
)

raw_clusters = kmeans_final.fit_predict(X_scaled)

joblib.dump(kmeans_final, "Stage1_KMeans_Model.pkl")

# ----------------------------------------------------------
# 8. PHYSICAL REORDERING OF CLUSTERS (CRITICAL STEP)
# ----------------------------------------------------------
# Clusters are reordered based on increasing CO2 affinity
# Criterion: mean (Coul+LJ)-LI-CO2 (more negative = stronger absorption)

df_tmp = df.copy()
df_tmp["RawCluster"] = raw_clusters

affinity_metric = (
    df_tmp.groupby("RawCluster")["Coul-LI-CO2"]
    .mean()
    +
    df_tmp.groupby("RawCluster")["LJ-LI-CO2"]
    .mean()
)

# Sort from weakest to strongest absorption
cluster_order = affinity_metric.sort_values(ascending=False).index.tolist()

# Deterministic physical mapping
cluster_mapping = {
    old_label: new_label
    for new_label, old_label in enumerate(cluster_order)
}

# Apply physical labels
df["Classe"] = [cluster_mapping[c] for c in raw_clusters]

# Save mapping explicitly (IMPORTANT)
joblib.dump(cluster_mapping, "Stage1_Cluster_Physical_Map.pkl")

# ----------------------------------------------------------
# 9. Diagnostics
# ----------------------------------------------------------

print("\nCluster physical ordering (diagnostic):")
for old, new in cluster_mapping.items():
    mean_dg = affinity_metric.loc[old]
    print(f"Raw cluster {old} → Classe {new} | mean (Coulomb+LJ)-LI-CO2 = {mean_dg:.3f}")

print("\nFinal class distribution:")
print(df["Classe"].value_counts().sort_index())

# ----------------------------------------------------------
# 10. Save classified dataset
# ----------------------------------------------------------

output_file = "CO2-Class-KMeans-Classified.xlsx"
df.to_excel(output_file, index=False)

print("\nStage 1 completed successfully.")
print(f"Classified data saved to: {output_file}")
print("Scaler, K-Means model, and physical cluster map saved.")

# ==========================================================
# STAGE 1 - Complement — Physically ordered cluster visualization
# ==========================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. Load the classified dataset: "CO2-Class-KMeans-Classified.xlsx"
# ----------------------------------------------------------

classified_file = "CO2-Class-KMeans-Classified.xlsx"
df = pd.read_excel(classified_file)

# Variables used in K-Means clustering
cluster_features = [
    "Coul-LI-CO2",
    "LJ-LI-CO2",
    "HB-LI-CO2",
    "Lifetime-LI-CO2",
    "DG-LI-CO2",
]

cluster_label = "Classe"

df_plot = df[cluster_features + [cluster_label]].copy()

# ----------------------------------------------------------
# 2. Seaborn style configuration
# ----------------------------------------------------------

sns.set_theme(
    style="whitegrid",
    context="notebook",
    font_scale=1.1
)

# ----------------------------------------------------------
# 3. Pairplot with physically meaningful color scale
# ----------------------------------------------------------

pairplot = sns.pairplot(
    data=df_plot,
    vars=cluster_features,
    hue=cluster_label,
    palette=sns.color_palette("plasma", df_plot[cluster_label].nunique()),
    diag_kind="hist",
    corner=False,
    height=2.6,
    plot_kws={
        "alpha": 0.75,
        "s": 45,
        "edgecolor": "k"
    }
)

# ----------------------------------------------------------
# 4. Figure adjustments
# ----------------------------------------------------------

pairplot.fig.suptitle(
    "Physically Ordered K-Means Clusters of LI–CO₂ Interaction Descriptors",
    y=1.02,
    fontsize=16
)

# ----------------------------------------------------------
# 5. Save and show
# ----------------------------------------------------------

output_fig = "Stage1_KMeans_PairPlot.png"
plt.savefig(output_fig, dpi=300, bbox_inches="tight")
plt.show()

print(f"Pairplot successfully generated and saved as: {output_fig}")

"""# **Part 02 - Artificial Neural Network for CO2 Aborption Prediciton**"""

# ==========================================================
# STAGE 2 — Multilayer Perceptron (MLP) for CO2 absorption
# classification with OVERSAMPLING and ORDINAL ANALYSIS
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error
)
from sklearn.utils import resample

# ----------------------------------------------------------
# 1. Load datasets for Artificial Neural Network training:
#     -- DATA-AAIL.xlsx
#     -- CO2-Class-KMeans-Classified.xlsx
# ----------------------------------------------------------

data_file  = "DATA-AAIL.xlsx"
class_file = "CO2-Class-KMeans-Classified.xlsx"

df_data  = pd.read_excel(data_file)
df_class = pd.read_excel(class_file)

print("Datasets loaded successfully.")

# ----------------------------------------------------------
# 2. Merge datasets (import PHYSICALLY ORDERED class from Stage 1)
# ----------------------------------------------------------

df = pd.merge(
    df_data,
    df_class[["Systems", "Classe"]],
    on="Systems",
    how="inner"
)

df = df.rename(columns={"Classe": "Class"})
print("Physical class column successfully imported from Stage 1.\n")

# ----------------------------------------------------------
# 3. Class interpretation (PHYSICALLY ORDERED, DATA-DRIVEN)
# ----------------------------------------------------------

unique_classes = np.sort(df["Class"].unique())

print("Class interpretation (from Stage 1 physical ordering):")

for c in unique_classes:
    if c == unique_classes.min():
        label = "Lowest relative CO₂ affinity"
    elif c == unique_classes.max():
        label = "Highest relative CO₂ affinity"
    else:
        label = "Intermediate relative CO₂ affinity"

    print(f"  Class {c} → {label}")

print(
    "\nNOTE: Class labels are ordinal and physically ordered.\n"
    "Their meaning is defined by the relative LI–CO₂ interaction strength\n"
    "(⟨E_Coul + E_LJ⟩) obtained in Stage 1, and NOT by fixed absolute categories.\n"
)

print(df.head())

# ----------------------------------------------------------
# 4. Define features and target
# ----------------------------------------------------------

feature_columns = [
    "C1", "C2",
    "TC", "TN", "TO", "TH",
    "MSD", "Dens",
    "Coul", "LJ",
    "HBs", "DHB", "LFTHB"
]

X = df[feature_columns].values
y = df["Class"].values

n_classes = len(np.unique(y))
print(f"Number of classes: {n_classes}\n")

# ----------------------------------------------------------
# 4.1 Define REGULARIZED MLP model
# ----------------------------------------------------------

# Relu 128 32 = ~70%
# Logistic 256, 128, 32 = ~67%

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 32),
    activation="relu",
    solver="adam",
    alpha=1e-5,
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    random_state=42,
    learning_rate="adaptive",
    learning_rate_init=0.001
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", mlp)
])

# ----------------------------------------------------------
# 5. Stratified K-Fold Cross Validation with OVERSAMPLING
# ----------------------------------------------------------

k_folds = 5
skf = StratifiedKFold(
    n_splits=k_folds,
    shuffle=True,
    random_state=42
)

all_true = []
all_pred = []

fold_accuracies = []
fold_f1 = []
fold_mae = []

print("Starting Stratified K-Fold Cross Validation...\n")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ------------------------------------------------------
    # Oversampling (TRAIN SET ONLY)
    # ------------------------------------------------------

    df_train = pd.DataFrame(X_train, columns=feature_columns)
    df_train["Class"] = y_train

    max_size = df_train["Class"].value_counts().max()

    df_balanced = []
    for cls in df_train["Class"].unique():
        df_cls = df_train[df_train["Class"] == cls]
        df_balanced.append(
            resample(
                df_cls,
                replace=True,
                n_samples=max_size,
                random_state=42
            )
        )

    df_balanced = pd.concat(df_balanced)

    X_train_bal = df_balanced[feature_columns].values
    y_train_bal = df_balanced["Class"].values

    # ------------------------------------------------------
    # Train and evaluate
    # ------------------------------------------------------

    pipeline.fit(X_train_bal, y_train_bal)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    mae = mean_absolute_error(y_test, y_pred)   # ORDINAL METRIC

    fold_accuracies.append(acc)
    fold_f1.append(f1)
    fold_mae.append(mae)

    all_true.extend(y_test)
    all_pred.extend(y_pred)

    print(
        f"Fold {fold}: "
        f"accuracy = {acc:.2f} | "
        f"macro-F1 = {f1:.2f} | "
        f"ordinal MAE = {mae:.2f}"
    )

print("\n---------------------------------------------")
print(f"Mean accuracy   : {np.mean(fold_accuracies):.2f}")
print(f"Mean macro-F1   : {np.mean(fold_f1):.2f}")
print(f"Mean ordinal MAE: {np.mean(fold_mae):.2f}")
print("---------------------------------------------")

# ----------------------------------------------------------
# 6. Confusion Matrix (normalized)
# ----------------------------------------------------------

cm = confusion_matrix(all_true, all_pred, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", values_format=".2f")
plt.title("Normalized Confusion Matrix — MLP (K-Fold CV)")
plt.show()

# ----------------------------------------------------------
# 7. Train final model on FULL dataset (with oversampling)
# ----------------------------------------------------------

df_full = pd.DataFrame(X, columns=feature_columns)
df_full["Class"] = y

max_size = df_full["Class"].value_counts().max()
df_full_bal = []

for cls in df_full["Class"].unique():
    df_cls = df_full[df_full["Class"] == cls]
    df_full_bal.append(
        resample(
            df_cls,
            replace=True,
            n_samples=max_size,
            random_state=42
        )
    )

df_full_bal = pd.concat(df_full_bal)

X_bal = df_full_bal[feature_columns].values
y_bal = df_full_bal["Class"].values

pipeline.fit(X_bal, y_bal)

joblib.dump(pipeline, "Stage2_MLP_CO2_Classifier.pkl")

print("\nFinal model trained on full balanced dataset.")
print("Model saved as: Stage2_MLP_CO2_Classifier.pkl")

# ----------------------------------------------------------
# 8. Interactive prediction mode
# ----------------------------------------------------------

print("\n--- Interactive Prediction Mode ---\n")
print("Provide the physicochemical properties of the new ionic liquid system.")
print("All values must be consistent with the training data domain.\n")

# ----------------------------------------------------------
# 8.0 Feature descriptions (single source of truth)
# ----------------------------------------------------------

feature_descriptions = {
    "C1": "Proportion of ionic liquid 1",
    "C2": "Proportion of ionic liquid 2",
    "TC": "Carbon content",
    "TN": "Nitrogen content",
    "TO": "Oxygen content",
    "TH": "Hydrogen content",
    "MSD": "Diffusion coefficient",
    "Dens": "Density of the ionic liquid mixture",
    "Coul": "Coulomb interaction energy (IL–IL)",
    "LJ": "Lennard-Jones interaction energy (IL–IL)",
    "HBs": "Number of hydrogen bonds",
    "DHB": "Hydrogen bond dissociation energy",
    "LFTHB": "Hydrogen bond lifetime"
}

# ----------------------------------------------------------
# 8.0.1 User input (STRICT training order)
# ----------------------------------------------------------

user_input = []

for name in feature_columns:
    desc = feature_descriptions.get(name, "No description available")
    value = float(input(f"{name} — {desc}: "))
    user_input.append(value)

user_input = np.array(user_input).reshape(1, -1)

# ----------------------------------------------------------
# 8.1 Domain consistency check (training-space statistics)
# ----------------------------------------------------------

print("\n--- Domain Consistency Check ---\n")

X_train = df[feature_columns].values
mean_vec = X_train.mean(axis=0)
std_vec = X_train.std(axis=0)

z_scores = np.abs((user_input.flatten() - mean_vec) / std_vec)

out_of_domain = []

for i, name in enumerate(feature_columns):
    if z_scores[i] > 3.0:
        out_of_domain.append(
            (name,
             user_input[0, i],
             mean_vec[i],
             std_vec[i],
             z_scores[i])
        )

if out_of_domain:
    print("⚠ WARNING: Potential extrapolation detected.\n")
    print("The following input features are outside the training domain:\n")
    print(f"{'Feature':<8} {'Value':>12} {'Mean':>12} {'Std':>12} {'|Z|':>8}")
    print("-" * 60)

    for name, val, mean, std, z in out_of_domain:
        print(f"{name:<8} {val:12.4e} {mean:12.4e} {std:12.4e} {z:8.2f}")

    print("\nPredictions involving extrapolation may have reduced reliability.\n")
else:
    print("✔ All input features are within the training data domain.\n")

# ----------------------------------------------------------
# 8.2 Prediction and probabilities
# ----------------------------------------------------------

predicted_class = int(pipeline.predict(user_input)[0])
predicted_proba = pipeline.predict_proba(user_input)[0]

n_classes = len(predicted_proba)

# ----------------------------------------------------------
# 8.3 Ordinal class meaning (AUTO-GENERATED, k-agnostic)
# ----------------------------------------------------------

cluster_meaning = {
    0: "Lowest CO₂ absorption regime",
    n_classes - 1: "Highest CO₂ absorption regime"
}

# Intermediate ordinal classes
for c in range(1, n_classes - 1):
    cluster_meaning[c] = f"Intermediate CO₂ absorption regime (level {c})"

# ----------------------------------------------------------
# 8.4 Ordinal confidence and reliability index
# ----------------------------------------------------------

# Expected ordinal value
ordinal_expectation = np.sum(
    np.arange(len(predicted_proba)) * predicted_proba
)

ordinal_deviation = abs(ordinal_expectation - predicted_class)

# Entropy-based confidence (normalized)
epsilon = 1e-12
entropy = -np.sum(predicted_proba * np.log(predicted_proba + epsilon))
max_entropy = np.log(len(predicted_proba))
confidence_index = 1.0 - entropy / max_entropy

# ----------------------------------------------------------
# 8.5 Output
# ----------------------------------------------------------

print("\n===================================")
print(f"Predicted class (ordinal): {predicted_class}")
print(f"Physical interpretation  : {cluster_meaning[predicted_class]}")
print("-----------------------------------")
print("Class probabilities:")

for cls, prob in enumerate(predicted_proba):
    meaning = cluster_meaning.get(cls, "Undefined")
    print(f"  Class {cls} ({meaning}): {prob:.3f}")

print("-----------------------------------")
print(f"Expected ordinal level     : {ordinal_expectation:.2f}")
print(f"Ordinal deviation (|Δ|)    : {ordinal_deviation:.2f}")
print(f"Confidence index (0%-100%)     : {confidence_index*100:.1f}%")

if confidence_index < 0.6:
    print("\n⚠ Low confidence prediction.")
    print("   Consider additional simulations or experimental validation.")
else:
    print("\n✔ Prediction is internally consistent and reliable.")

print("===================================")

"""# **Part 3 - Physical Interpretation and Prediction**"""

# ==========================================================
# STAGE 3 — Cluster-based interpretation with uncertainty
# and probability-weighted estimates (ADJUSTED VERSION)
# ==========================================================
# IMPORTANT:
# This stage must be executed AFTER Stage 2
# in the same session, or with predicted_class
# and predicted_proba explicitly loaded.
# ==========================================================

import pandas as pd
import numpy as np

# ----------------------------------------------------------
# 1. Load classified dataset from Stage 1
# ----------------------------------------------------------

cluster_file = "CO2-Class-KMeans-Classified.xlsx"
df = pd.read_excel(cluster_file)

interaction_features = [
    "Coul-LI-CO2",
    "LJ-LI-CO2",
    "HB-LI-CO2",
    "Lifetime-LI-CO2",
    "DG-LI-CO2"
]

cluster_label = "Classe"

# ----------------------------------------------------------
# 2. Compute cluster statistics and uncertainties
# ----------------------------------------------------------

cluster_stats = {}

for cid in sorted(df[cluster_label].unique()):

    cluster_data = df[df[cluster_label] == cid]
    stats = {}

    for feat in interaction_features:
        values = cluster_data[feat].dropna().values

        stats[feat] = {
            "mean": np.mean(values),
            "std": np.std(values, ddof=1),
            "min": np.min(values),
            "max": np.max(values),
            "p05": np.percentile(values, 5),
            "p95": np.percentile(values, 95),
            "n": len(values)
        }

    cluster_stats[cid] = stats

print("Cluster statistics and uncertainties computed successfully.")

# ----------------------------------------------------------
# 3. INPUT from Stage 2
# ----------------------------------------------------------
# These variables MUST be available from Stage 2:
# predicted_class
# predicted_proba

predicted_cluster = int(predicted_class)
cluster_probabilities = np.array(predicted_proba)

# ----------------------------------------------------------
# 4. Ordinal confidence diagnostics
# ----------------------------------------------------------

ordinal_expectation = np.sum(
    np.arange(len(cluster_probabilities)) * cluster_probabilities
)
ordinal_deviation = abs(ordinal_expectation - predicted_cluster)

epsilon = 1e-12
entropy = -np.sum(cluster_probabilities * np.log(cluster_probabilities + epsilon))
max_entropy = np.log(len(cluster_probabilities))
confidence_entropy = 1.0 - entropy / max_entropy

# ----------------------------------------------------------
# 5. Cluster-conditional report (dominant regime)
# ----------------------------------------------------------

print("\n====================================================")
print(f"Most probable CO₂ absorption cluster: {predicted_cluster}")
print("Cluster-conditional LI–CO₂ descriptors")
print("====================================================\n")

for feat, s in cluster_stats[predicted_cluster].items():
    print(f"{feat}:")
    print(f"  Mean ± std            : {s['mean']:.4f} ± {s['std']:.4f}")
    print(f"  Robust interval (5–95): [{s['p05']:.4f}, {s['p95']:.4f}]")
    print(f"  Full observed range   : [{s['min']:.4f}, {s['max']:.4f}]")
    print(f"  Sample size           : n = {s['n']}\n")

# ----------------------------------------------------------
# 6. Probability-weighted estimates (across regimes)
# ----------------------------------------------------------

print("====================================================")
print("Probability-weighted LI–CO₂ descriptors")
print("(accounts for regime overlap and NN uncertainty)")
print("====================================================\n")

for feat in interaction_features:

    weighted_mean = 0.0
    second_moment = 0.0

    for cid, prob in zip(sorted(cluster_stats.keys()), cluster_probabilities):

        mu = cluster_stats[cid][feat]["mean"]
        sigma = cluster_stats[cid][feat]["std"]

        weighted_mean += prob * mu
        second_moment += prob * (sigma**2 + mu**2)

    weighted_std = np.sqrt(max(second_moment - weighted_mean**2, 0.0))

    print(f"{feat}:")
    print(f"  Weighted mean        : {weighted_mean:.4f}")
    print(f"  Effective uncertainty: ± {weighted_std:.4f}\n")

# ----------------------------------------------------------
# 7. Global reliability assessment
# ----------------------------------------------------------

print("----------------------------------------------------")
print("Global reliability assessment")
print("----------------------------------------------------")
print(f"Entropy-based confidence : {confidence_entropy:.3f}")
print(f"Ordinal deviation (|Δ|)  : {ordinal_deviation:.2f}")

if confidence_entropy > 0.75 and ordinal_deviation < 0.4:
    print("Interpretation: HIGH confidence prediction.")
elif confidence_entropy > 0.50 and ordinal_deviation < 0.8:
    print("Interpretation: MODERATE confidence prediction.")
else:
    print("Interpretation: LOW confidence (significant regime overlap).")

print("----------------------------------------------------")