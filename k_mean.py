from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st
import random

# Markdown สำหรับเขียนหัวข้อ ใช้แทน function st.write
st.write("""
# k-means Clustering

By using [sklearn](https://scikit-learn.org/stable/) and
[streamlit](https://streamlit.io/) library

Create simple webapp with python script !!
""")

# สร้างตัวแปรสุ่มใช้สำหรับการสุ่มข้อมูล
random_point = {'state':random.randint(0,100)}

# สร้าง sidebar
if st.sidebar.button('Random data'):
    random_point['state'] = random.randint(0,100)

st.sidebar.write('State = ', random_point['state'])
cluster_std = st.sidebar.slider('dispersion', 0.2, 3.0, 0.2, 0.2)
n_clusters = st.sidebar.selectbox('Number of clusters', range(1,10))


# สร้างข้อมูล 2 มิติ
x, y = make_blobs(n_samples=200, n_features=2, centers=5, cluster_std=cluster_std,random_state=random_point['state'])

# สร้าง k-means
kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=10, max_iter=300)
y_kmeans = kmeans.fit_predict(x)

# Plot
fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(x[:, 0], x[:, 1], s=100, c=kmeans.labels_, cmap='Set1')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=400, marker='*', color='k')
st.pyplot(fig)