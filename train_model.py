import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# =========================================
# 1. LOAD DATA
# =========================================
books = pd.read_csv("Books.csv")
ratings = pd.read_csv("Ratings.csv")
users = pd.read_csv("Users.csv")

# Standardize column names
books.columns = books.columns.str.lower().str.replace("-", "_")
ratings.columns = ratings.columns.str.lower().str.replace("-", "_")
users.columns = users.columns.str.lower().str.replace("-", "_")

# =========================================
# 2. BASIC EDA & CLEANING
# =========================================
# Remove invalid ratings
ratings = ratings[ratings["book_rating"] > 0]

print("Total ratings after cleaning:", len(ratings))

# =========================================
# 3. FEATURE ENGINEERING (IMPORTANT)
# =========================================
# Keep only active users
active_users = ratings["user_id"].value_counts()
active_users = active_users[active_users >= 50].index
ratings = ratings[ratings["user_id"].isin(active_users)]

# Keep only popular books
popular_books = ratings["isbn"].value_counts()
popular_books = popular_books[popular_books >= 50].index
ratings = ratings[ratings["isbn"].isin(popular_books)]

print("Ratings after filtering:", len(ratings))

# =========================================
# 4. MERGE METADATA (KEEP IMAGE LINKS)
# =========================================
final_df = ratings.merge(
    books[["isbn", "book_title", "book_author", "image_url_m"]],
    on="isbn",
    how="left"
)

# =========================================
# 5. USER–BOOK MATRIX
# =========================================
book_pivot = final_df.pivot_table(
    index="book_title",
    columns="user_id",
    values="book_rating"
).fillna(0)

print("Pivot shape:", book_pivot.shape)

# =========================================
# 6. MODEL: COSINE SIMILARITY
# =========================================
similarity_scores = cosine_similarity(book_pivot)

# =========================================
# 7. POPULARITY MODEL (COLD START)
# =========================================
popular_books_df = (
    final_df.groupby("isbn")
    .count()["book_rating"]
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
    .merge(books, on="isbn")
)

# =========================================
# 8. SAVE ARTIFACTS
# =========================================
pickle.dump(book_pivot, open("book_pivot.pkl", "wb"))
pickle.dump(similarity_scores, open("similarity_scores.pkl", "wb"))
pickle.dump(books, open("books_metadata.pkl", "wb"))
pickle.dump(popular_books_df, open("popular_books.pkl", "wb"))

print("✅ Training completed successfully (Stable model saved)")
