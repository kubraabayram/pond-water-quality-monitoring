import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

DATA_PATH = Path("data/pond_iot_2023.csv")
MODEL_PATH = Path("models/pond_model.joblib")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def rule_score(pH, TDS, temp):
    if 6.8 <= pH <= 8.2:
        pH_score = 2
    elif 6.0 <= pH < 6.8 or 8.2 < pH <= 8.6:
        pH_score = 1
    else:
        pH_score = 0

    if TDS <= 450:
        tds_score = 2
    elif 450 < TDS <= 600:
        tds_score = 1
    else:
        tds_score = 0

    if 23 <= temp <= 28:
        temp_score = 2
    elif 20 <= temp < 23 or 28 < temp <= 32:
        temp_score = 1
    else:
        temp_score = 0

    total = pH_score + tds_score + temp_score
    if total >= 5:
        return 2
    elif total >= 3:
        return 1
    return 0

df = pd.read_csv(DATA_PATH)
df = df[["water_pH", "TDS", "water_temp"]]
df["label"] = df.apply(lambda r: rule_score(r["water_pH"], r["TDS"], r["water_temp"]), axis=1)

# Artificial Bad class generation
if (df["label"] == 0).sum() < 20:
    bad_samples = pd.DataFrame({
        "water_pH": np.random.uniform(4.5, 6.0, 80),
        "TDS": np.random.uniform(700, 1500, 80),
        "water_temp": np.random.uniform(33, 40, 80),
        "label": 0
    })
    df = pd.concat([df, bad_samples], ignore_index=True)

X = df[["water_pH", "TDS", "water_temp"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.joblib")

# Models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "SVM": SVC(kernel="rbf"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LogisticRegression": LogisticRegression(max_iter=500)
}

scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    scores[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="macro"),
        "Recall": recall_score(y_test, pred, average="macro"),
        "F1 Score": f1_score(y_test, pred, average="macro")
    }

plt.figure(figsize=(8, 5))
plt.bar(scores.keys(), [s["Accuracy"] for s in scores.values()])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.savefig(RESULTS_DIR / "accuracy.png")
plt.close()

# Best model selection
best_model_name = max(scores, key=lambda m: scores[m]["Accuracy"])
best_model = models[best_model_name]
joblib.dump(best_model, MODEL_PATH)

pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix – {best_model_name}")
plt.colorbar()

labels = ["Bad", "Moderate", "Good"]
ticks = np.arange(len(labels))
plt.xticks(ticks, labels)
plt.yticks(ticks, labels)

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(RESULTS_DIR / "cm.png")
plt.close()

pd.DataFrame(scores).to_csv(RESULTS_DIR / "model_scores.csv")

print("\nTraining completed!")
print(f"Best model: {best_model_name}")
