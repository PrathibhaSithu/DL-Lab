#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv('Au_nanoparticle_dataset.csv')

df.head()


# In[2]:


# Filter the dataframe to include only the specified columns
filtered_df = df[['N_total', 'N_bulk', 'N_surface', 'R_avg']]

# Display the first 20 samples of this dataframe
filtered_df.head(20)


# In[3]:


# Calculate the mean, standard deviation, and quartile values for each feature
mean_values = filtered_df.mean()
std_values = filtered_df.std()
quartiles = filtered_df.quantile([0.25, 0.5, 0.75])

print("Mean values:\n", mean_values)
print("\nStandard Deviation values:\n", std_values)
print("\nQuartile values:\n", quartiles)


# In[4]:


import matplotlib.pyplot as plt

# Create a 1x4 layout for histograms
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Plot histograms for each feature
features = ['N_total', 'N_bulk', 'N_surface', 'R_avg']
for i, feature in enumerate(features):
    axes[i].hist(filtered_df[feature], bins=30, alpha=0.7, color='blue')
    axes[i].set_title(f'Histogram of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[5]:


import seaborn as sns

# Visualize scatter plots and histograms using seaborn's pairplot
sns.pairplot(filtered_df)
plt.show()


# In[6]:


# Custom PairGrid
g = sns.PairGrid(filtered_df)

# Change the diagonal plots to include histograms and KDE plots
g.map_diag(sns.histplot, kde=True)

# Change the lower half to bivariate KDE plots
g.map_lower(sns.kdeplot)

# Change the upper half to bivariate histograms
g.map_upper(sns.histplot, bins=30)

plt.show()


# In[ ]:




