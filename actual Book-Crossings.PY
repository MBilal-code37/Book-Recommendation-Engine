import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load the Book-Crossings dataset
books = pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1', error_bad_lines=False)
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1', error_bad_lines=False)
users = pd.read_csv('BX-Users.csv', sep=';', encoding='latin-1', error_bad_lines=False)

def preprocess_ratings(ratings):
    """
    Filter users with less than 200 ratings and books with less than 100 ratings
    """
    # Filter users with at least 200 ratings
    user_ratings_count = ratings['User-ID'].value_counts()
    active_users = user_ratings_count[user_ratings_count >= 200].index
    ratings_filtered = ratings[ratings['User-ID'].isin(active_users)]
    
    # Filter books with at least 100 ratings
    book_ratings_count = ratings_filtered['ISBN'].value_counts()
    popular_books = book_ratings_count[book_ratings_count >= 100].index
    ratings_filtered = ratings_filtered[ratings_filtered['ISBN'].isin(popular_books)]
    
    return ratings_filtered

def create_user_book_matrix(ratings_filtered, books):
    """
    Create a user-book rating matrix
    """
    # Merge with book titles
    ratings_with_titles = ratings_filtered.merge(
        books[['ISBN', 'Book-Title']], 
        on='ISBN', 
        how='left'
    )
    
    # Create pivot table
    book_matrix = ratings_with_titles.pivot_table(
        index='Book-Title',
        columns='User-ID',
        values='Book-Rating'
    ).fillna(0)
    
    return book_matrix

def create_knn_model(book_matrix):
    """
    Create and train the KNN model
    """
    # Convert to sparse matrix
    book_matrix_sparse = csr_matrix(book_matrix.values)
    
    # Train KNN model
    model = NearestNeighbors(
        metric='cosine',
        algorithm='brute',
        n_neighbors=6
    )
    model.fit(book_matrix_sparse)
    
    return model

def get_recommends(book_title):
    """
    Return 5 book recommendations with distances
    """
    global model, book_matrix
    
    try:
        # Find the book index
        book_idx = list(book_matrix.index).index(book_title)
        
        # Get nearest neighbors
        distances, indices = model.kneighbors(
            book_matrix.iloc[book_idx, :].values.reshape(1, -1),
            n_neighbors=6
        )
        
        # Create recommendations list
        recommendations = []
        for i in range(1, 6):
            rec_book = book_matrix.index[indices.flatten()[i]]
            rec_distance = distances.flatten()[i]
            recommendations.append([rec_book, rec_distance])
        
        # Sort by distance
        recommendations.sort(key=lambda x: x[1])
        
        return [book_title, recommendations]
    
    except ValueError:
        return [book_title, []]

# Process the data
print("Processing ratings data...")
ratings_filtered = preprocess_ratings(ratings)

print("Creating user-book matrix...")
book_matrix = create_user_book_matrix(ratings_filtered, books)

print("Training KNN model...")
model = create_knn_model(book_matrix)

print("Model is ready!")

# Test the function
test_result = get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
print(test_result)
