import numpy as np 
from scipy.cluster.hierarchy import dendrogram, linkage 
import matplotlib.pyplot as plt 
  
# randomly chosen dataset 
X = np.array([[1, 2], [1, 4], [1, 0], 
              [4, 2], [4, 4], [4, 0]]) 
  
# Perform hierarchical clustering 
Z = linkage(X, 'ward') 
  
# Plot dendrogram 
dendrogram(Z) 
  
plt.title('Hierarchical Clustering Dendrogram') 
plt.xlabel('Data point') 
plt.ylabel('Distance') 
plt.show()