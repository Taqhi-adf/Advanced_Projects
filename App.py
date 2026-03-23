# Streamlit Web App: Iris Flower Classification using Decision Tree

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1️⃣ Page Configuration
# ----------------------------
st.set_page_config(page_title="Iris Flower Classification", layout="wide")

st.title("🌸 Iris Flower Classification App")
st.write("""
This app predicts the **species of an Iris flower** based on its features  
using a **Decision Tree Classifier**.
""")

# 2️⃣ Load Dataset
# ----------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

df = pd.concat([X, y], axis=1)
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# 3️⃣ Sidebar – User Inputs
# ----------------------------
st.sidebar.header("🔧 Model & Parameters")

criterion = st.sidebar.selectbox("Select Criterion", ("gini", "entropy"))
test_size = st.sidebar.slider("Test Size (for splitting)", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random State", min_value=0, max_value=100, value=42)

st.sidebar.markdown("---")
st.sidebar.header("🌺 Enter Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# 4️⃣ EDA Section
# ----------------------------
st.subheader("📊 Dataset Overview")
st.dataframe(df.head())

col1, col2 = st.columns(2)
with col1:
    st.write("### Class Distribution")
    st.bar_chart(df['species_name'].value_counts())

with col2:
    st.write("### Feature Statistics")
    st.dataframe(df.describe())

# 5️⃣ Model Training
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=random_state
)

model = DecisionTreeClassifier(criterion=criterion, random_state=random_state)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)


# 6️⃣ Model Evaluation
# ----------------------------
st.subheader("🤖 Model Evaluation")
st.write(f"**Accuracy:** {acc:.4f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names, ax=ax)
st.pyplot(fig)

# 7️⃣ Decision Tree Visualization
# ----------------------------
st.subheader("🌲 Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(18, 10))
plot_tree(model, filled=True, feature_names=iris.feature_names,
          class_names=iris.target_names, rounded=True, fontsize=10)
st.pyplot(fig)

# 8️⃣ Prediction Section
# ----------------------------
st.subheader("🔮 Predict Flower Species")

sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(sample)[0]
predicted_species = iris.target_names[prediction]

st.success(f"### ✅ Predicted Species: **{predicted_species.capitalize()}**")

st.write("Input values used for prediction:")
st.write({
    "Sepal Length": sepal_length,
    "Sepal Width": sepal_width,
    "Petal Length": petal_length,
    "Petal Width": petal_width,
})

# 9️⃣ Bonus – Pairplot Visualization
# ----------------------------
with st.expander("📈 Explore Feature Relationships (Pairplot)"):
    st.write("This visualization shows how features differ among flower species.")
    fig2 = sns.pairplot(df, hue="species_name", corner=True)
    st.pyplot(fig2)

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
---
**Developed for:** AICTE Approved Internship – Data Science  
**Goal:** Predict Iris flower species using Decision Tree Classifier  
**Libraries Used:** scikit-learn, pandas, seaborn, matplotlib, streamlit  
""")
