from flask import Flask, render_template,request
import pandas as pd
import pickle
import difflib
import numpy as np


popular_df = pd.read_pickle('popular.pkl')
pt = pd.read_pickle('pt.pkl')
similarity_score = pd.read_pickle('similarity_score.pkl')
books = pd.read_pickle('books.pkl')

books_df = pd.read_pickle("books_df.pkl")
tfidf = pd.read_pickle("tfidf_vectorizer.pkl")
tfidf_matrix = pd.read_pickle("tfidf_matrix.pkl")
nn = pd.read_pickle("nearest_neighbors.pkl")
book_similarity = pd.read_pickle("books_similarity.pkl")


def content_recommend(book_title, top_n=5):
    book_title = book_title.strip().lower()

    # Find the closest matching book title
    close_matches = difflib.get_close_matches(book_title, books_df["title"].str.lower(), n=1, cutoff=0.1)

    if not close_matches:
        return []

    matched_book = close_matches[0]

    # Get the index of the matched book
    index = books_df[books_df["title"].str.lower() == matched_book].index[0]

    # Find similar books using nearest neighbors
    distances, indices = nn.kneighbors(tfidf_matrix[index], n_neighbors=top_n + 1)

    # Return only book titles
    recommended_books = books_df.iloc[indices[0]]["title"].tolist()
    return [matched_book] + recommended_books[1:]  # Include the searched book first


def collab_recommend(book_name, top_n=5):
    book_name = book_name.strip().lower()
    close_matches = difflib.get_close_matches(book_name, book_similarity.index, n=1, cutoff=0.1)

    if not close_matches:
        return []

    matched_book = close_matches[0]
    if matched_book not in book_similarity.index:
        return []

    # Get Top Similar Books
    similar_books = book_similarity[matched_book].sort_values(ascending=False)[1:top_n+1]

    return list(similar_books.index)


def hybrid_recommend(book_title):
    book_title = book_title.strip()  # Remove extra spaces

    content_books = content_recommend(book_title)
    collab_books = collab_recommend(book_title)

    if content_books or collab_books:
        final_titles = list(dict.fromkeys([book_title] + content_books + collab_books))

        books_df["normalized_title"] = books_df["title"].str.strip().str.lower()

        final_recommendations = books_df[books_df["normalized_title"].isin([title.strip().lower() for title in final_titles])]

        final_recommendations = final_recommendations.drop_duplicates(subset=["title"])

        final_list = final_recommendations[["title", "author", "rating", "language", "likedPercent"]].values.tolist()

        final_list = sorted(final_list, key=lambda x: x[0].strip().lower() != book_title.strip().lower())

        return final_list
    else:
        return []


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['Num_Ratings'].values),
                           rating=list(popular_df['AVG_Ratings'].values))


@app.route('/search')
def recommendation_ui():
    return render_template('recommendation.html')

# @app.route('/recommendation',methods=['POST'])
# def recommendation():
#     user_input = request.form.get('user_input')
#     close_matches = difflib.get_close_matches(user_input, pt.index, n=1, cutoff=0.1)
#
#     if not close_matches:
#         print("No book found with a name similar to:", user_input)
#         return
#     matched_book = close_matches[0]
#
#     index = np.where(pt.index == matched_book)[0][0]
#     similar_books = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[0:9]
#
#     data = []
#     for i in similar_books:
#         item = []
#         temp_df = books[books['Book-Title'] == pt.index[i[0]]]
#         item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
#         item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
#         item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
#         data.append(item)
#     return render_template('recommendation.html',data=data)



@app.route('/recommendation',methods=['POST'])
def recommendation():
    user_input = request.form.get('user_input').strip()

    recommended_books = hybrid_recommend(user_input)

    return render_template('recommendation.html', data=recommended_books)
@app.route('/about')
def about():
    return  render_template('about.html')




if __name__ == '__main__':
    app.run(debug=True)
