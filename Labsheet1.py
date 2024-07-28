#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv(Au_nanoparticle_dataset.csv)


# In[4]:


df = pd.read_csv('Au_nanoparticle_dataset.csv')


# In[5]:


df.head()


# In[6]:


import numpy as np


# In[8]:


random_array = np.random.exponential(scale=scale_parameter, size=(4, 4))

print(random_array)


# In[9]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


exponential_array = np.random.exponential(scale=scale_parameter, size=(100000, 1))

print(exponential_array)


# In[11]:


exponential_data=np.random.exponential(scale=1.0, size=(100000,1))

print(exponential_data)


# In[12]:


uniform_data=np.random.uniform(low = 0.0, high = 1.0, size = (100000,1))


# In[13]:


plt.hist(exponential_data, density=True, bins=100, histtype="step", color="green", label="exponential")


# In[14]:


plt.hist(uniform_data, density=True, bins=100, histtype="step", color="blue", label="uniform")


# In[26]:


normal_data=np.random.normal(loc=0.0, scale=1.0, size=(100000,1))


# In[16]:


plt.hist(normal_data, density=True, bins=100, histtype="step", color="blue", label="normal")


# In[27]:


plt.hist(exponential_data, density=True, bins=100, histtype="step", color="green", label="exponential")
plt.hist(uniform_data, density=True, bins=100, histtype="step", color="blue", label="uniform")
plt.hist(normal_data, density=True, bins=100, histtype="step", color="red", label="normal")

# Adjust plot settings for better visualization

plt.legend(loc="upper right")
plt.title("Random distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()


# In[ ]:





# In[28]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate X and Y values
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Calculate Z values
Z = X**2 + Y**2

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D plot of Z = X^2 + Y^2')

plt.show()


# In[29]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'HP': np.random.randint(50, 150, 100),
    'Attack': np.random.randint(50, 150, 100),
    'Defense': np.random.randint(50, 150, 100),
    'Sp. Atk': np.random.randint(50, 150, 100),
    'Sp. Def': np.random.randint(50, 150, 100),
    'Speed': np.random.randint(50, 150, 100)
})

# Calculate Pearson correlation
pearson_corr = data.corr(method='pearson')

# Calculate Spearman rank correlation
spearman_corr = data.corr(method='spearman')

# Plot heatmaps
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Pearson correlation heatmap
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', ax=ax[0])
ax[0].set_title('Pearson Correlation')

# Spearman rank correlation heatmap
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', ax=ax[1])
ax[1].set_title('Spearman Rank Correlation')

plt.show()


# In[ ]:




