# Task 12: KMeans – Customer Segmentation

## 📌 Overview
This project applies **KMeans Clustering** to segment customers based on their **Annual Income** and **Spending Score** using the Mall Customer dataset.  
The goal is to identify meaningful customer groups to help businesses design **targeted marketing strategies**.

This task demonstrates the complete **unsupervised learning workflow** from data preprocessing to business interpretation.

---

## 🛠 Tools & Technologies
- Python  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 📊 Dataset
**Mall Customer Segmentation Dataset (Kaggle)**  

Key features:
- `Annual Income (k$)`
- `Spending Score (1-100)`

Column `CustomerID` is removed as it has no predictive value.

---

## 📂 Project Structure
task-12-kmeans-customer-segmentation/
│
├── data/raw/Mall_Customers.csv
├── data/processed/mall_customers_segmented.csv
├── notebooks/Task12_KMeans_Customer_Segmentation.ipynb
├── visuals/elbow_plot.png
├── visuals/clusters.png
├── reports/Task12_Cluster_Insights.pdf
├── README.md
└── requirements.txt


---

## 🔹 Step 1: Load Dataset
```python
import pandas as pd
df = pd.read_csv("Mall_Customers.csv")
🔹 Step 2: Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
🔹 Step 3: Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
🔹 Step 4: Elbow Method
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1,11), inertia, marker='o')
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
🔹 Step 5: Train Final KMeans Model
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
🔹 Step 6: Cluster Visualization
import seaborn as sns

sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='Set1'
)
plt.title("Customer Segments")
plt.show()
🔹 Step 7: Save Segmented Dataset
df.to_csv("mall_customers_segmented.csv", index=False)
📊 Example Cluster Interpretation
Cluster	Description
0	High Income – High Spenders (Premium)
1	Low Income – High Spenders
2	High Income – Low Spenders
3	Low Income – Low Spenders
4	Average Income – Average Spending
🎯 Final Outcome
After completing this task, the intern:

Understands unsupervised clustering

Can choose optimal K using Elbow Method

Can visualize and interpret clusters

Can apply segmentation to real business problems

