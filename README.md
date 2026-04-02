# 🌸 Iris Flower Classification using Decision Tree

## 📌 Overview

This project demonstrates a complete **machine learning workflow** using the famous Iris dataset.
It builds and evaluates **Decision Tree models (Gini & Entropy)**, performs data visualization, and predicts flower species.

The goal is to classify iris flowers into:

* Setosa
* Versicolor
* Virginica

based on their physical features and real variables.

## 📊 Dataset

* Source: Built-in dataset from Scikit-learn
* Records: 150 samples
* Features:

  * Sepal Length
  * Sepal Width
  * Petal Length
  * Petal Width
* Target:

  * Species (3 classes)

## 🛠️ Tools & Technologies

* Python 🐍
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn

## 🔄 Project Workflow (Step-by-Step)

### 1️⃣ Load Dataset

* Imported dataset using Scikit-learn
* Converted into Pandas DataFrame for analysis

```python
from sklearn.datasets import load_iris
iris = load_iris()
```

### 2️⃣ Data Exploration

* Checked dataset structure
* Viewed sample records
* Analyzed class distribution

### 3️⃣ Data Visualization

* Used Seaborn Pairplot
* Visualized relationships between features
* Observed class separability

```python
sns.pairplot(df, hue='species_name')
```

### 4️⃣ Train-Test Split

* Split data into training (80%) and testing (20%)
* Used stratified sampling for balanced classes

```python
train_test_split(X, y, test_size=0.2, stratify=y)
```

### 5️⃣ Model Training

#### 🔹 Decision Tree (Gini Index)

* Uses Gini impurity
* Faster computation

#### 🔹 Decision Tree (Entropy)

* Uses Information Gain
* More informative splits

```python
DecisionTreeClassifier(criterion='gini')
DecisionTreeClassifier(criterion='entropy')
```

### 6️⃣ Model Evaluation

* Accuracy Score
* Classification Report
* Confusion Matrix

```python
accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)
```

### 7️⃣ Confusion Matrix Visualization

* Used heatmap for better understanding
* Compared predicted vs actual values

```python
sns.heatmap(cm, annot=True)
``

### 8️⃣ Decision Tree Visualization

* Plotted tree structure
* Displayed decision rules graphically

```python
plot_tree(model, filled=True)
```

### 9️⃣ Extract Decision Rules

* Converted tree into readable text format

```python
export_text(model)
```
### 🔟 Prediction on New Data

* Tested model with new flower input
* Predicted species successfully

```python
model.predict([[5.1, 3.5, 1.4, 0.2]])
```
## 📈 Results

* Both models performed with high accuracy
* Entropy-based tree provided slightly better interpretability
* Clear class separation observed in visualization

## 📊 Sample Output
Accuracy: 1.0000
Prediction: setosa

## 📁 Project Structure

```
📁 iris-decision-tree
│
├── iris_decision_tree.py
├── README.md
```

## 🚀 How to Run the Project

### 1. Clone Repository

```bash
git clone https://github.com/your-username/iris-decision-tree.git
cd iris-decision-tree
```
## 2. Install Dependencies

```bash
pip install scikit-learn pandas matplotlib seaborn graphviz
```

### 3. Run the Script

```bash
python iris_decision_tree.py
```

## 💡 Key Learnings

* Decision Tree algorithms (Gini vs Entropy)
* Model evaluation techniques
* Data visualization using Seaborn
* Feature importance understanding
* Real-world ML workflow

## 🚀 Future Improvements

* Hyperparameter tuning (max_depth, pruning)
* Cross-validation
* Deploy using Streamlit
* Compare with other models (SVM, Random Forest)

## 👨‍💻 Author

**Taqhi Ma**
Machine Learning Enthusiast & Researcher 🚀

