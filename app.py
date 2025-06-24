import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SETUP: Load models and data ONCE at startup ---
print("Flask app starting... attempting to load model and data.")

# Build absolute paths to your files from the script's location
# This is the most important part for a successful deployment on Render.
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'book_recommender.h5')
    ENCODER_PATH = os.path.join(BASE_DIR, 'book_encoder.pkl')
    BOOKS_CSV_PATH = os.path.join(BASE_DIR, 'books.csv')

    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load the book encoder
    with open(ENCODER_PATH, 'rb') as f:
        book_encoder = pickle.load(f)

    # Load the books metadata
    books_df = pd.read_csv(BOOKS_CSV_PATH, sep=';', on_bad_lines='skip', encoding='latin-1')
    books_df.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']

    # Extract the book embedding weights once for efficiency
    book_embedding_weights = model.get_layer('BookEmbedding').get_weights()[0]

    print("✅ Model and data loaded successfully! The server is ready.")

except Exception as e:
    # Print the exact error to the Render logs if something fails during startup
    print(f"❌ An error occurred during file loading: {e}")
    # Raising the error ensures the app stops if loading fails, which helps in debugging
    raise e


# --- 2. The Recommendation Function ---
def get_api_recommendations(book_title, num_recs=5):
    """
    Finds books similar to a given title using the learned embeddings
    and returns their full details in a structured format.
    """
    try:
        # Convert book title to its integer ID
        book_id_encoded = book_encoder.transform([book_title])[0]
        # Get the embedding vector for our target book
        book_vec = book_embedding_weights[book_id_encoded].reshape(1, -1)
    except ValueError:
        # If the book title is not in the encoder, return an error message
        return {"error": f"Book '{book_title}' not found in the dataset."}

    # Calculate cosine similarity between this book and all other books
    similarities = cosine_similarity(book_vec, book_embedding_weights)[0]
    
    # Get the indices of the books with the highest similarity scores
    similar_book_indices = np.argsort(similarities)[::-1]

    recommendations = []
    rec_count = 0
    for idx in similar_book_indices:
        # Skip the book itself (it will always have a similarity of 1.0)
        if idx == book_id_encoded:
            continue
        
        # Convert the book index back to its title
        recommended_title = book_encoder.inverse_transform([idx])[0]
        similarity_score = float(similarities[idx])
        
        # Look up the book details in the original books DataFrame
        book_details = books_df[books_df['Book-Title'] == recommended_title].drop_duplicates('Book-Title')
        
        if not book_details.empty:
            # Create a dictionary for the recommended book's details
            rec_data = {
                "title": recommended_title,
                "author": book_details['Book-Author'].values[0],
                "year": str(book_details['Year-Of-Publication'].values[0]),
                "publisher": book_details['Publisher'].values[0],
                "image_url": book_details['Image-URL-M'].values[0],
                "similarity_score": round(similarity_score, 4)
            }
            recommendations.append(rec_data)
        
        rec_count += 1
        # Stop after we have found the desired number of recommendations
        if rec_count >= num_recs:
            break
            
    return {"recommendations": recommendations}


# --- 3. The Flask API Endpoints ---
app = Flask(__name__)

@app.route('/')
def index():
    """A simple landing page to confirm the API is running."""
    return "<h1>Book Recommendation API</h1><p>Use the /recommend endpoint. Example: /recommend?title=The+Da+Vinci+Code</p>"

@app.route('/recommend', methods=['GET'])
def recommend():
    """The main endpoint to get book recommendations."""
    # Get the 'title' from the URL query parameters (e.g., ?title=...)
    book_title = request.args.get('title')
    
    # If no title is provided, return a helpful error message
    if not book_title:
        return jsonify({"error": "Please provide a 'title' query parameter."}), 400
        
    # Get the recommendation results
    results = get_api_recommendations(book_title)
    
    # Return the results as a JSON response
    return jsonify(results)

# This part is optional for Render but good for local testing.
# Gunicorn will run the 'app' object directly.
if __name__ == '__main__':
    # When running locally, start the app in debug mode
    app.run(debug=True, port=5000)