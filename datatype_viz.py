import pandas as pd
import matplotlib.pyplot as plt

# Load your file
df = pd.read_csv("C:/Users/User/Desktop/2026 Spring/IDX exchange DA/Week 2/Week 2-3 Outputs_20260409_144110/column_dtype_summary.csv")

# Correct filtering (case-sensitive match)
listing_df = df[df['dataset'] == 'Listing']
sold_df = df[df['dataset'] == 'Sold']

# Count data types
listing_counts = listing_df['dtype'].value_counts()
sold_counts = sold_df['dtype'].value_counts()

# Debug check (important)
print("Listing counts:\n", listing_counts)
print("\nSold counts:\n", sold_counts)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].pie(
    listing_counts,
    labels=listing_counts.index,
    autopct='%1.1f%%'
)
axes[0].set_title('Listing Data Types')

axes[1].pie(
    sold_counts,
    labels=sold_counts.index,
    autopct='%1.1f%%'
)
axes[1].set_title('Sold Data Types')

plt.tight_layout()
plt.show()

axes[0].set_title('Listing Data Types', y=0.95)
axes[1].set_title('Sold Data Types', y=0.95)

plt.tight_layout(rect=[0, 0, 1, 0.95])