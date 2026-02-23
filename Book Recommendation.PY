import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load the data (already provided in the notebook)
books = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv?raw=true')
ratings = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv?raw=true')

# For Book-Crossings dataset (if using the actual dataset)
# books = pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1', error_bad_lines=False)
# ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1', error_bad_lines=False)

def preprocess_data(ratings, books):
    """
    Preprocess the ratings data by filtering users with < 200 ratings
    and books with < 100 ratings
    """
    # Count ratings per user and filter
    user_rating_count = ratings['user_id'].value_counts()
    users_to_keep = user_rating_count[user_rating_count >= 200].index
    ratings_filtered = ratings[ratings['user_id'].isin(users_to_keep)]
    
    # Count ratings per book and filter
    book_rating_count = ratings_filtered['book_id'].value_counts()
    books_to_keep = book_rating_count[book_rating_count >= 100].index
    ratings_filtered = ratings_filtered[ratings_filtered['book_id'].isin(books_to_keep)]
    
    return ratings_filtered

def create_book_matrix(ratings_filtered, books):
    """
    Create a user-book matrix for KNN
    """
    # Merge ratings with book titles
    ratings_with_titles = ratings_filtered.merge(books[['book_id', 'title']], on='book_id')
    
    # Create a pivot table (users as rows, books as columns)
    book_matrix = ratings_with_titles.pivot_table(
        index='title', 
        columns='user_id', 
        values='rating'
    ).fillna(0)
    
    return book_matrix

def train_knn_model(book_matrix):
    """
    Train the KNN model using cosine distance
    """
    # Convert to sparse matrix for efficiency
    book_matrix_sparse = csr_matrix(book_matrix.values)
    
    # Train KNN model
    model_knn = NearestNeighbors(
        metric='cosine', 
        algorithm='brute', 
        n_neighbors=6  # 5 recommendations + the book itself
    )
    model_knn.fit(book_matrix_sparse)
    
    return model_knn, book_matrix

def get_recommends(book_title):
    """
    Get book recommendations based on a given book title
    
    Args:
        book_title (str): Title of the book to find recommendations for
        
    Returns:
        list: Formatted list with book title and 5 recommendations with distances
    """
    global model_knn, book_matrix
    
    try:
        # Find the book in our matrix
        book_index = list(book_matrix.index).index(book_title)
        
        # Get distances and indices of nearest neighbors
        distances, indices = model_knn.kneighbors(
            book_matrix.iloc[book_index, :].values.reshape(1, -1), 
            n_neighbors=6
        )
        
        # Create recommendations list
        recommendations = []
        for i in range(1, len(distances.flatten())):
            recommended_book = book_matrix.index[indices.flatten()[i]]
            distance = distances.flatten()[i]
            recommendations.append([recommended_book, distance])
        
        # Sort by distance (closest first)
        recommendations.sort(key=lambda x: x[1])
        
        # Format the return value as specified
        return [book_title, recommendations]
        
    except ValueError:
        return [book_title, f"Book '{book_title}' not found in the dataset."]

# Main execution
if __name__ == "__main__":
    # Preprocess the data
    print("Preprocessing data...")
    ratings_filtered = preprocess_data(ratings, books)
    
    # Create book matrix
    print("Creating book matrix...")
    book_matrix = create_book_matrix(ratings_filtered, books)
    
    # Train KNN model
    print("Training KNN model...")
    model_knn, book_matrix = train_knn_model(book_matrix)
    
    print("Model ready for recommendations!")
    
    # Test the function
    test_book = "The Queen of the Damned (Vampire Chronicles (Paperback))"
    recommendations = get_recommends(test_book)
    print(recommendations)
