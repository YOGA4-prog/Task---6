# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For pairplot and histplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap

# 1. Load the Iris dataset
url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
data = pd.read_csv(url)
print("First 5 rows of the dataset:")
print(data.head())

# Feature matrix (X) and target vector (y)
X = data.drop(columns=['species'])
y = data['species']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 2. Experiment with different K values
k_range = range(1, 11)
accuracies = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"K={k} -> Accuracy: {acc:.2f}")

# --- 1. Accuracy vs K Plot ---
plt.figure(figsize=(8, 4))
plt.plot(k_range, accuracies, marker='o', linestyle='-')
plt.title("KNN Accuracy for Different K Values")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# 3. Evaluate Best K
best_k = k_range[np.argmax(accuracies)]
print(f"\nBest K value: {best_k}")

# Confusion Matrix
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_best_pred = knn_best.predict(X_test)
cm = confusion_matrix(y_test, y_best_pred, labels=knn_best.classes_)

# --- 2. Confusion Matrix Plot ---
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_best.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (K={best_k})")
plt.show()

# 4. Decision Boundary Visualization (only with 2 features)
print("\nVisualizing decision boundaries using 2 features...")
X_vis = data[['petal_length', 'petal_width']]
y_vis = y
X_vis_scaled = scaler.fit_transform(X_vis)

Xv_train, Xv_test, yv_train, yv_test = train_test_split(X_vis_scaled, y_vis, test_size=0.3, random_state=42)

knn_vis = KNeighborsClassifier(n_neighbors=best_k)
knn_vis.fit(Xv_train, yv_train)

h = 0.02  # step size in the mesh
x_min, x_max = X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1
y_min, y_max = X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = pd.factorize(Z)[0]  # Convert to numerical
Z = Z.reshape(xx.shape)

# --- 3. Decision Boundary Plot ---
plt.figure(figsize=(8, 6))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.contourf(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_vis_scaled[:, 0], X_vis_scaled[:, 1], c=pd.factorize(y_vis)[0], cmap=cmap_bold, edgecolor='k', s=20)
plt.title(f"Decision Boundary using Petal Length & Width (K={best_k})")
plt.xlabel("Petal Length (scaled)")
plt.ylabel("Petal Width (scaled)")
plt.show()

# --- 4. Pair Plot (optional for EDA) ---
sns.pairplot(data, hue="species")
plt.suptitle("Pair Plot of All Features by Species", y=1.02)
plt.show()

# --- 5. Feature Distribution Histogram ---
sns.histplot(data=data, x="petal_length", hue="species", kde=True)
plt.title("Petal Length Distribution by Species")
plt.show()
