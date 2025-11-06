#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[5]:


#model = pickle.load(open('Book_Recmd.pkl','rb'))
#model


# In[7]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix

st.set_page_config(page_title="Book Recommendation", layout="wide")
st.title("📚 Book Recommendation using Hierarchical Clustering")

# Load data
df_path = "User_Rating_Books.csv"  # Adjust this path if n|eeded
try:
    df = pd.read_csv(df_path)
    st.success("Data loaded successfully!")
    st.write(df.head())
except FileNotFoundError:
    st.error(f"File not found at: {df_path}")
    st.stop()

# Filter: Users & Books with enough ratings
df["Num_Ratings_By_User"] = df.groupby("user_id")["book_rating"].transform("count")
df["Num_Ratings_By_Book"] = df.groupby("book_title")["book_rating"].transform("count")
filtered_df = df[(df['Num_Ratings_By_User'] >= 20) & (df['Num_Ratings_By_Book'] >= 20)]

# Reduce dataset for memory management
top_users = filtered_df['user_id'].value_counts().head(500).index
top_books = filtered_df['book_title'].value_counts().head(500).index
filtered_df = filtered_df[filtered_df['user_id'].isin(top_users) & filtered_df['book_title'].isin(top_books)]

# Remove duplicates by averaging
filtered_df = (
    filtered_df.groupby(['book_title', 'user_id'], as_index=False)['book_rating'].mean()
)

# Pivot and convert to sparse matrix
pivot_table = filtered_df.pivot(index='book_title', columns='user_id', values='book_rating')
sparse_matrix = csr_matrix(pivot_table.fillna(0).values)


# Optional: Visualize most rated books
st.subheader("📊 Top 10 Most Rated Books")
top_rated = filtered_df['book_title'].value_counts().head(10)
fig, ax = plt.subplots()
sns.barplot(x=top_rated.values, y=top_rated.index, ax=ax)
st.pyplot(fig)

# Scale using MaxAbsScaler
scaler = MaxAbsScaler()
scaled_matrix = scaler.fit_transform(sparse_matrix)

# Reduce dimensions using TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=42)
reduced_matrix = svd.fit_transform(scaled_matrix)

# Hierarchical clustering
st.subheader("🔗 Hierarchical Clustering")
cluster_range = st.slider("Select number of clusters:", 2, 15, 5)
model = AgglomerativeClustering(n_clusters=cluster_range, linkage='ward')
labels = model.fit_predict(reduced_matrix)
sil_score = silhouette_score(reduced_matrix, labels)
st.write(f"📈 Silhouette Score for {cluster_range} clusters: **{sil_score:.4f}**")

# Add cluster labels to pivot_table
pivot_table['Cluster'] = labels

# Book recommendation section
st.subheader("🎯 Book Recommendation Based on Clusters")
book_choice = st.selectbox("Select a book:", pivot_table.index)

if book_choice:
    selected_cluster = pivot_table.loc[book_choice, 'Cluster']
    similar_books = pivot_table[pivot_table['Cluster'] == selected_cluster].index.tolist()
    similar_books.remove(book_choice)

    st.markdown(f"Books in the same cluster as **{book_choice}**:")
    for book in similar_books[:10]:
        st.write("📖", book)


# In[ ]:





# In[ ]:





# In[43]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




