"""K-MEANS CUSTOMER SEGMENTATION"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np

print("Starting K-Means Customer Segmentation...")
print("=" * 50)

np.random.seed(42)

centers = [
    [-3, 9],   # Young High Spenders
    [-7, -7],  # Young Low Spenders  
    [5, 2],    # Older Moderate Spenders
    [-9, 7]    # Very Young High Spenders
]

X, _ = make_blobs(n_samples=300, centers=centers, cluster_std=1.0, random_state=42)

print("CUSTOMER SEGMENTATION")
print(f"Number of customers: {X.shape[0]}")
print(f"Features: Age Score (X-axis), Spending Score (Y-axis)")
print()

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_

print("CLUSTER PROFILES:")
print("=" * 40)

cluster_names = ["Young Big Spenders", "Young Budget-Conscious", 
                 "Older Moderate", "Very Young Luxury"]

for i in range(4):
    age, spend = kmeans.cluster_centers_[i]
    size = sum(labels == i)
    
    age_desc = "Young" if age < -2 else "Older" if age > 2 else "Average"
    spend_desc = "High Spender" if spend > 4 else "Low Spender" if spend < -4 else "Moderate"
    
    print(f"\nCluster {i} ({cluster_names[i]}):")
    print(f"   Customers: {size}")
    print(f"   Profile: {age_desc}, {spend_desc}")
    print(f"   Center: Age={age:.1f}, Spend={spend:.1f}")

plt.figure(figsize=(12, 8))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
markers = ['o', 's', '^', 'D']

for i in range(4):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], 
                c=colors[i], marker=markers[i], 
                label=cluster_names[i], alpha=0.7, s=100, 
                edgecolors='white', linewidth=0.5)
    
    plt.scatter(kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1],
                marker='X', s=300, c='black', edgecolors='white', 
                linewidth=2, zorder=10)
    
    plt.text(kmeans.cluster_centers_[i, 0] + 0.3, kmeans.cluster_centers_[i, 1] + 0.3,
             f'Cluster {i}', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))

plt.xlabel('Age Score (Younger ← → Older)', fontsize=13)
plt.ylabel('Spending Score (Low ← → High)', fontsize=13)
plt.title('Customer Segmentation using K-Means Clustering', 
          fontsize=16, fontweight='bold', pad=20)

plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')

plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

plt.text(-10, 9, 'Young High Spenders', fontsize=10, alpha=0.8)
plt.text(-10, -8, 'Young Low Spenders', fontsize=10, alpha=0.8)
plt.text(3, 9, 'Older High Spenders', fontsize=10, alpha=0.8)
plt.text(3, -8, 'Older Low Spenders', fontsize=10, alpha=0.8)

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("CUSTOMER PREDICTION DEMO")
print("=" * 50)

sample_customers = [
    {"name": "Student", "age": -6.0, "spend": -5.0},
    {"name": "Young Professional", "age": -4.0, "spend": 7.0},
    {"name": "Family Person", "age": 4.0, "spend": 2.0},
    {"name": "Teenager", "age": -8.0, "spend": 8.0},
    {"name": "Senior", "age": 6.0, "spend": 1.0}
]

print("\nPredicting customer segments:")
for customer in sample_customers:
    features = [[customer["age"], customer["spend"]]]
    cluster = kmeans.predict(features)[0]
    
    print(f"\n{customer['name']}:")
    print(f"   Age Score: {customer['age']:.1f}")
    print(f"   Spending Score: {customer['spend']:.1f}")
    print(f"   Segment: {cluster_names[cluster]}")

print("\n" + "=" * 50)
print("ADDITIONAL ANALYSIS")
print("=" * 50)

total_customers = len(X)
print("\nCluster Distribution:")
for i in range(4):
    percentage = (sum(labels == i) / total_customers) * 100
    print(f"{cluster_names[i]}: {percentage:.1f}% of customers")

plt.savefig('customer_segmentation.png', dpi=120, bbox_inches='tight')
print(f"\nPlot saved as 'customer_segmentation.png'")

print("\n" + "=" * 50)
print("DEMO COMPLETE!")
print("=" * 50)