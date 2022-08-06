from sklearn.cluster import KMeans
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
df = pd.read_csv("final_gravity.csv")
radius = df.iloc[: , 3]
mass = df.iloc[: , 2]
print(radius , mass)

X = df.iloc[:,[3,4]]

print(X)
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i , init = "k-means++" , random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#plt.figure(figsize = (10 , 6))
#sns.lineplot(range(1,11) , wcss , marker = "o" , color = "red")
#plt.show()

##number of clusters are 3
##scaling the data 
scaler = MinMaxScaler()
scaler.fit(df[["Star_Radiuses"]])
df["Star_Radiuses"] = scaler.transform(df[["Star_Radiuses"]])
scaler.fit(df[["Star_Mass"]])
df["Star_Mass"] = scaler.transform(df[["Star_Mass"]])

km = KMeans(n_clusters = 3 ,init = "k-means++" , random_state = 42 )
y_pred = km.fit_predict(df[["Star_Mass"]])
df["clusters"] = y_pred

df1 = df[df.clusters == 1]
df2 = df[df.clusters == 2]
df3 = df[df.clusters == 3]
plt.scatter(df1["Star_Radiuses"] , df1["Star_Mass"], color = "green" , label = "cluster-1")
plt.scatter(df2["Star_Radiuses"],df2["Star_Mass"], color = "red" , label = "cluster-2")
plt.scatter(df3["Star_Radiuses"], df3["Star_Mass"] , color = "blue" , label = "cluster-3")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,0],color = "purple",marker = "*" , label = "centroids")
plt.legend()
plt.show()