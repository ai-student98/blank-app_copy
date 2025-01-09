import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report

# Generate mock crime dataset
np.random.seed(42)
n_samples = 1000
sample_df = pd.DataFrame({
    'Hour': np.random.randint(0, 24, size=n_samples),
    'AREA NAME': np.random.choice(['Area A', 'Area B', 'Area C'], size=n_samples),
    'Crm Cd Desc': np.random.choice(['Theft', 'Assault', 'Burglary', 'Vandalism'], size=n_samples),
    'Vict Sex': np.random.choice(['M', 'F'], size=n_samples),
    'Vict Descent': np.random.choice(['Asian', 'Hispanic', 'White', 'Black'], size=n_samples),
    'Premis Desc': np.random.choice(['Street', 'Residence', 'Commercial', 'Park'], size=n_samples),
    'Weapon Desc': np.random.choice(['Gun', 'Knife', 'None'], size=n_samples),
    'Status Desc': np.random.choice(['Open', 'Closed'], size=n_samples),
    'DayOfWeek': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], size=n_samples),
    'TimeOfDay': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], size=n_samples)
})

sample_df['Hour_Sin'] = np.sin(2 * np.pi * sample_df['Hour'] / 24)
sample_df['Hour_Cos'] = np.cos(2 * np.pi * sample_df['Hour'] / 24)

categorical_columns = ['AREA NAME', 'Crm Cd Desc', 'Vict Sex', 'Vict Descent', 'Premis Desc', 
                       'Weapon Desc', 'Status Desc', 'DayOfWeek', 'TimeOfDay']

X_encoded = pd.get_dummies(sample_df.drop(columns=['Hour']), columns=categorical_columns, drop_first=True)
X_encoded['Hour_Sin'] = sample_df['Hour_Sin']
X_encoded['Hour_Cos'] = sample_df['Hour_Cos']
X_encoded.columns = X_encoded.columns.str.replace(r'[\[\], ]+', '_', regex=True)
X_encoded.columns = X_encoded.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
X_encoded.columns = X_encoded.columns.str.strip('_')

bins = [0, 6, 12, 18, 24]
labels = ['Night', 'Morning', 'Afternoon', 'Evening']
sample_df['TimeOfDay'] = pd.cut(sample_df['Hour'], bins=bins, labels=labels, right=False)
y_hour = sample_df['TimeOfDay']

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_hour, test_size=0.2, random_state=42)

st.title('Crime Data Model Comparison')
st.subheader('Correlation Matrix')
plt.figure(figsize=(12, 10))
corr_matrix = X_encoded.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', cbar=True)
st.pyplot(plt)

n_estimators = st.sidebar.slider('n_estimators (Random Forest)', min_value=10, max_value=200, value=100, step=10)
max_depth = st.sidebar.slider('max_depth (Random Forest)', min_value=5, max_value=50, value=None, step=5)
learning_rate = st.sidebar.slider('learning_rate (XGBoost, LightGBM)', min_value=0.01, max_value=0.3, value=0.1, step=0.01)

models = {
    'RandomForest': RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', learning_rate=learning_rate),
    'LightGBM': LGBMClassifier(random_state=42, learning_rate=learning_rate),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2 Score': r2}
    st.write(f"{name} - MSE: {mse:.4f}, R2 Score: {r2:.4f}")
    st.subheader(f'{name} - Confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    plt.clf()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)
    st.write(f"Classification Report for {name}:")
    st.text(classification_report(y_test, y_pred))

results_df = pd.DataFrame(results).T
st.subheader('Model Performance Comparison')
st.dataframe(results_df)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
results_df['MSE'].plot(kind='bar', ax=axes[0], color='skyblue', title='Model Comparison: MSE')
axes[0].set_ylabel('Mean Squared Error')
results_df['R2 Score'].plot(kind='bar', ax=axes[1], color='salmon', title='Model Comparison: R2 Score')
axes[1].set_ylabel('R2 Score')
st.pyplot(fig)

for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        st.subheader(f'Top 10 Feature Importances - {name}')
        feature_importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
        top_features = feature_importances.nlargest(10)
        plt.clf()
        top_features.plot(kind='barh', figsize=(8, 5), color='teal')
        plt.title(f'Top 10 Feature Importances - {name}')
        plt.xlabel('Importance Score')
        st.pyplot(plt)
