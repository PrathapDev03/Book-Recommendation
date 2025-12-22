import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------------------
# LOAD SAVED FILES (COSINE SIMILARITY MODEL)
# -----------------------------------------
book_pivot = pickle.load(open("book_pivot.pkl", "rb"))
similarity_scores = pickle.load(open("similarity_scores.pkl", "rb"))
books = pickle.load(open("books_metadata.pkl", "rb"))
popular_books = pickle.load(open("popular_books.pkl", "rb"))

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# -----------------------------------------
# CUSTOM CSS (PROFESSIONAL LOOK)
# -----------------------------------------
st.markdown("""
<style>
h1 {
    font-size: 42px;
}
.book-title {
    font-size: 14px;
    font-weight: 600;
}
.book-author {
    font-size: 12px;
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# HEADER
# -----------------------------------------
st.title("📚 Book Recommendation System")
st.write(
    "Type a book name to get **Top 8 similar book recommendations**. "
    "You can also **copy and share** the recommendations easily."
)

# -----------------------------------------
# INPUT
# -----------------------------------------
book_name = st.text_input(
    "🔍 Enter Book Name",
    placeholder="Example: Harry Potter and the Sorcerer's Stone"
)

TOP_N = 8

# -----------------------------------------
# RECOMMENDATION FUNCTION
# -----------------------------------------
def recommend_books(book_name, top_n=8):
    # Cold-start → show popular books
    if book_name not in book_pivot.index:
        return popular_books.head(top_n)

    index = np.where(book_pivot.index == book_name)[0][0]

    similar_items = sorted(
        list(enumerate(similarity_scores[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:top_n + 1]

    recommendations = []
    for i in similar_items:
        temp = books[books["book_title"] == book_pivot.index[i[0]]]
        if not temp.empty:
            recommendations.append(temp.iloc[0])

    return pd.DataFrame(recommendations)

# -----------------------------------------
# DISPLAY RESULTS
# -----------------------------------------
if st.button("📖 Recommend Books"):
    results = recommend_books(book_name, TOP_N)

    st.subheader(f"✨ Top {TOP_N} Recommended Books")

    # ---------- IMAGE GRID ----------
    cols = st.columns(4)
    for idx, row in results.iterrows():
        with cols[idx % 4]:
            st.image(row["image_url_m"], width=180)
            st.markdown(
                f"<div class='book-title'>{row['book_title']}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='book-author'>{row['book_author']}</div>",
                unsafe_allow_html=True
            )

    # ---------- COPY & SHARE LIST ----------
    st.markdown("---")
    st.subheader("📋 Copy & Share Recommendations")

    copy_text = ""
    for i, row in enumerate(results.itertuples(), start=1):
        copy_text += f"{i}. {row.book_title} — {row.book_author}\n"

    st.text_area(
        "You can copy this list and share it anywhere 👇",
        copy_text,
        height=220
    )
