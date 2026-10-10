import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

#loading dataset
print("STEP 1: Loading the dataset")
CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(CSV_PATH)

#loading details
print(f"✔  Loaded {df.shape[0]} rows and {df.shape[1]} columns.\n")
print("Columns in the dataset:")
print(list(df.columns))
print()

#cleaning dataset
print("STEP 2: Cleaning the data")

df.drop(columns=["customerID"], inplace=True)
print("Dropped 'customerID' column.")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
 
num_bad = df["TotalCharges"].isna().sum()
print(f"Converted 'TotalCharges' to numeric. Found {num_bad} blank rows → dropping them.")
 
df.dropna(subset=["TotalCharges"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"   Dataset size after dropping: {df.shape[0]} rows.\n")

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print("Encoded target column 'Churn':  Yes → 1,  No → 0")
print(f"   Class distribution:\n{df['Churn'].value_counts().rename({1:'Churn (Yes)', 0:'No Churn (No)'})}\n")

binary_columns = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]
 
for col in binary_columns:
    df[col] = df[col].map({"Yes": 1, "No": 0})
print(f"Encoded binary (Yes/No) columns: {binary_columns}\n")

df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
print("Encoded 'gender':  Male → 1,  Female → 0\n")

multi_category_columns = [
    "MultipleLines",       # No phone service / No / Yes
    "InternetService",     # DSL / Fiber optic / No
    "OnlineSecurity",      # No internet service / No / Yes
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",            # Month-to-month / One year / Two year
    "PaymentMethod",       # 4 payment methods
]
df = pd.get_dummies(df, columns=multi_category_columns, drop_first=True)
print(f" One-Hot Encoded {len(multi_category_columns)} multi-category columns.")

print(f"   Dataset now has {df.shape[1]} columns after encoding.\n")

remaining_nulls = df.isnull().sum().sum()
print(f"Missing values remaining: {remaining_nulls}  (should be 0)\n")

print(f"Final dataset shape: {df.shape}\n")

#splitting data into training and testing datasets
print("STEP 3: Splitting data into Train and Test sets")
X = df.drop(columns=["Churn"])   # Everything except Churn as in input features
y = df["Churn"]                  # Only Churn as in what we want to predict
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
print(f"Training set:  {X_train.shape[0]} rows  ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Testing set:   {X_test.shape[0]} rows  ({X_test.shape[0]/len(X)*100:.1f}%)\n")

#training dataet
print("STEP 4: Training the Random Forest model")
rf_model = RandomForestClassifier(
    n_estimators=100,      # Build 100 decision trees
    max_depth=None,        # Let trees grow until data is pure (no limit)
    min_samples_split=2,   # A node needs at least 2 samples to be split
    min_samples_leaf=1,    # Each leaf must have at least 1 sample
    random_state=42,       # Reproducibility
    n_jobs=-1,             # Use all CPU cores for speed
)
rf_model.fit(X_train, y_train)
print("Training complete!\n")

#evaluating model
print("STEP 5: Evaluating the model on the Test set")
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy:  {accuracy * 100:.2f}%")
print("   (Percentage of customers correctly classified)\n")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("── Confusion Matrix ──")
print("Rows = Actual labels, Columns = Predicted labels")
print(f"  True Negatives  (correctly said No Churn): {cm[0][0]}")
print(f"  False Positives (wrongly said Churn):      {cm[0][1]}")
print(f"  False Negatives (missed actual Churn):     {cm[1][0]}")
print(f"  True Positives  (correctly said Churn):    {cm[1][1]}")
y_prob = rf_model.predict_proba(X_test)[:, 1]  # Get churn probabilities
auc_score = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {auc_score:.3f}")

#feature importance
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns,
).sort_values(ascending=False)

print("Top 15 most important features:\n")
print(feature_importance.head(15).to_string())
print()

#visualization
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Random Forest", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

fig, ax = plt.subplots(figsize=(9, 6))
feature_importance.head(15).plot(
    kind="barh",
    ax=ax,
    color="steelblue",
    edgecolor="white",
)
ax.invert_yaxis()   # Most important feature at the top
ax.set_title("Top 15 Feature Importances — Random Forest", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("Saved: feature_importance.png\n")

#summary
print(f"  Dataset rows used    : {len(df)}")
print(f"  Number of features   : {X.shape[1]}")
print(f"  Training samples     : {X_train.shape[0]}")
print(f"  Test samples         : {X_test.shape[0]}")
print(f"  Model accuracy       : {accuracy * 100:.2f}%")
print()
print("Output files saved in the same folder:")
print("  • confusion_matrix.png  — Visual of correct vs wrong predictions")
print("  • feature_importance.png — Which features matter most")