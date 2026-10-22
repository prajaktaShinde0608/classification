# Classification Algorithms and Comparative Analysis on the Iris dataset
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)

# Step 1: Load the dataset
iris = load_iris()
X, y = iris.data, iris.target
print("Samples:", X.shape[0], "| Features:", X.shape[1], "| Classes:", len(set(y)))

# Step 2: Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Define the models
models = {
    "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": GaussianNB(),
}

results = []

# Step 4: Train, predict and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    loss = log_loss(y_test, y_prob, labels=model.classes_)

    results.append(
        {
            "Algorithm": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Log Loss": loss,
        }
    )

    print("\n" + "=" * 45)
    print(name.upper(), "RESULTS")
    print("=" * 45)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"Log Loss  : {loss:.4f}")

# Step 5: Comparative table
df = pd.DataFrame(results)
print("\n\n" + "=" * 75)
print("COMPARATIVE ANALYSIS OF CLASSIFICATION ALGORITHMS")
print("=" * 75)
print(df.round(4).to_string(index=False))

# Step 6: Best model per metric
print("\n\nBEST MODEL ACCORDING TO EACH METRIC")
print("-" * 45)
print("Highest Accuracy :", df.loc[df["Accuracy"].idxmax(), "Algorithm"])
print("Highest Precision:", df.loc[df["Precision"].idxmax(), "Algorithm"])
print("Highest Recall   :", df.loc[df["Recall"].idxmax(), "Algorithm"])
print("Highest F1-Score :", df.loc[df["F1-Score"].idxmax(), "Algorithm"])
print("Lowest Log Loss  :", df.loc[df["Log Loss"].idxmin(), "Algorithm"])

# Step 7: Interpretation
best_f1 = df.loc[df["F1-Score"].idxmax()]
best_loss = df.loc[df["Log Loss"].idxmin()]
print("\n\nCOMPARISON ANALYSIS")
print("-" * 45)
print(f"{best_f1['Algorithm']} achieved the highest weighted F1-score of {best_f1['F1-Score']:.4f}.")
print(f"{best_loss['Algorithm']} achieved the lowest log loss of {best_loss['Log Loss']:.4f}.")
