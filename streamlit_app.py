import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1) Create sample dataset
data = {
    "Hour": [1,2,3,4,5,1,2,3,9,4,3,2,2,1],  # Extra rows so each 'Hour' has at least 2 items
    "AREA NAME": ["A","B","A","C","A","B","C","C","A","B","B","C","A","A"],
    "Crm Cd Desc": ["Desc1","Desc2","Desc1","Desc2","Desc1","Desc2","Desc1","Desc2","Desc1","Desc2","Desc2","Desc2","Desc1","Desc1"],
    "Vict Sex": ["M","F","M","F","M","F","M","F","M","F","F","F","M","M"],
    "Vict Descent": ["W","H","B","A","W","H","B","A","W","H","A","B","W","B"],
    "Premis Desc": ["House","Street","House","Street","House","Street","House","Street","House","Street","Street","House","Street","House"],
    "Weapon Desc": ["Gun","Knife","Gun","Knife","Gun","Knife","Gun","Knife","Gun","Knife","Knife","Knife","Gun","Gun"],
    "Status Desc": ["Open","Open","Open","Open","Open","Open","Open","Open","Open","Open","Open","Open","Open","Open"],
    "DayOfWeek": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun","Mon","Tue","Wed","Fri","Sat","Sun","Mon"],
    "TimeOfDay": ["Night","Night","Day","Day","Night","Night","Day","Day","Night","Night","Day","Night","Night","Night"]
}
sample_df = pd.DataFrame(data)

# 2) Convert Hour to categorical for classification tasks
sample_df["Hour"] = sample_df["Hour"].astype("category")

# 3) Build cyclical features for Hour
sample_df["Hour_Sin"] = np.sin(2*np.pi*sample_df["Hour"].astype(int)/24)
sample_df["Hour_Cos"] = np.cos(2*np.pi*sample_df["Hour"].astype(int)/24)

# 4) Prepare features (X) via one-hot encoding of categorical columns
categorical_columns = ["AREA NAME","Crm Cd Desc","Vict Sex","Vict Descent","Premis Desc","Weapon Desc","Status Desc","DayOfWeek","TimeOfDay"]
X_encoded = pd.get_dummies(
    sample_df.drop(columns=["Hour"]),
    columns=categorical_columns,
    drop_first=True
)
X_encoded.columns = X_encoded.columns.str.replace(r"[\[\], ]+", "_", regex=True)
X_encoded.columns = X_encoded.columns.str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
X_encoded.columns = X_encoded.columns.str.strip("_")

# 5) Target (y)
y_hour = sample_df["Hour"]  # Categorized hour

# 6) Split data
# NOTE: If you have classes with only 1 item, DO NOT USE stratify. 
# It will raise "The least populated class in y has only 1 member..."
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_hour, 
    test_size=0.2,
    random_state=42  # no stratify here
)

# 7) Create Streamlit UI
st.title("Crime Data Model Comparison")

st.subheader("Correlation Matrix")
plt.figure(figsize=(12, 10))
corr_matrix = X_encoded.corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", cbar=True)
st.pyplot(plt)

# 8) Sidebar hyperparameters
n_estimators = st.sidebar.slider("n_estimators (Random Forest)", 10, 200, 100, 10)
max_depth = st.sidebar.slider("max_depth (Random Forest)", 5, 50, 10, 5)
learning_rate = st.sidebar.slider("learning_rate (XGBoost, LightGBM)", 0.01, 0.3, 0.1, 0.01)

# 9) Models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss", learning_rate=learning_rate),
    "LightGBM": LGBMClassifier(random_state=42, learning_rate=learning_rate)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # For classification, MSE and R2 are somewhat less common metrics, but we'll do it anyway:
    # Convert categories to int so that MSE and R2 make sense
    mse = mean_squared_error(y_test.astype(int), y_pred.astype(int))
    r2 = r2_score(y_test.astype(int), y_pred.astype(int))
    results[name] = {"MSE": mse, "R2 Score": r2}

    st.write(f"{name} - MSE: {mse:.4f}, R2 Score: {r2:.4f}")

    st.subheader(f"{name} - Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)

    st.write(f"Classification Report for {name}:")
    st.text(classification_report(y_test, y_pred))

results_df = pd.DataFrame(results).T
st.subheader("Model Performance Comparison")
st.dataframe(results_df)

fig, axes = plt.subplots(1, 2, figsize=(15,6))
results_df["MSE"].plot(kind="bar", ax=axes[0], color="skyblue", title="Model Comparison: MSE")
axes[0].set_ylabel("Mean Squared Error")
results_df["R2 Score"].plot(kind="bar", ax=axes[1], color="salmon", title="Model Comparison: R2 Score")
axes[1].set_ylabel("R2 Score")
st.pyplot(fig)

for name,model in models.items():
    if hasattr(model, "feature_importances_"):
        st.subheader(f"Top 10 Feature Importances - {name}")
        feature_importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
        top_features = feature_importances.nlargest(10)
        top_features.plot(kind="barh", figsize=(8,5), color="teal")
        plt.title(f"Top 10 Feature Importances - {name}")
        plt.xlabel("Importance Score")
        st.pyplot(plt)
