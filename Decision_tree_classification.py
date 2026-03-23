# Decision Tree–based Iris Flower Classification with visualization ):

# iris_decision_tree.py
# =====================
# Run this file: python iris_decision_tree.py
# Required installations:
# pip install scikit-learn pandas matplotlib seaborn graphviz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️⃣ Load Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

# Combine for easy exploration
df = pd.concat([X, y], axis=1)
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# 2️⃣ Explore Data
print("=== Dataset Information ===")
print(df.head(), "\n")
print("Class Distribution:\n", df['species_name'].value_counts(), "\n")

# 3️⃣ Visualize Relationships
sns.pairplot(df, hue='species_name')
plt.suptitle("Iris Dataset Feature Relationships", y=1.02)
plt.show()

# 4️⃣ Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train,'\n')
print(y_train)

# 5️⃣ Train Decision Tree (Gini)
clf_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
clf_gini.fit(X_train, y_train)

# 6️⃣ Train Decision Tree (Entropy)
clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_entropy.fit(X_train, y_train)

# 7️⃣ Evaluate Both Models
for name, model in [("Gini", clf_gini), ("Entropy", clf_entropy)]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"=== Decision Tree ({name}) ===")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Greens', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title(f"Confusion Matrix ({name})")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # 8️⃣ Visualize the Best Model Tree
best_model = clf_entropy  # you can choose clf_gini instead
plt.figure(figsize=(16, 10))
plot_tree(
    best_model,
    filled=True,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Visualization (Entropy)")
plt.show()

# 9️⃣ Print Tree Rules in Text Format
print("\n=== Decision Tree Rules (Entropy) ===")
print(export_text(best_model, feature_names=iris.feature_names))

# 🔟 Predict a New Flower
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example input
pred = best_model.predict(sample)[0]
print("\nPrediction for sample [5.1, 3.5, 1.4, 0.2]:", iris.target_names[pred])

# the end...