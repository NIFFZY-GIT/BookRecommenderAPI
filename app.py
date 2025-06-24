import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SETUP: Load models and data ONCE at startup ---
print("Loading model and data... This may take a moment.")

# Get the directory of the current script to build absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the required files
MODEL_PATH = os.path.join(BASE_DIR, 'book_recommender.h5')
ENCODER_PATH = os.path.join(BASE_DIR, 'book_encoder.pkl')
BOOKS_CSV_PATH = os.path.join(BASE_DIR, 'books.csv') # Use lowercase to be safe

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except OSError:
    # A fallback for Render's specific environment if the above fails
    model = tf.keras.models.load_model('book_recommender.h5')

# Load the book encoder
with open(ENCODER_PATH, 'rb') as f:
    book_encoder = pickle.load(f)

# Load the books metadata
books_df = pd.read_csv(BOOKS_CSV_PATH, sep=';', on_bad_lines='skip', encoding='latin-1')
books_df.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']

# Extract the book embedding weights once
book_embedding_weights = model.get_layer('BookEmbedding').get_weights()[0]

print("✅ Model and data loaded successfully!")


# --- 2. The Recommendation Function ---
def get_api_recommendations(book_title, num_recs=5):
    try:
        book_id_encoded = book_encoder.transform([book_title])[0]
        book_vec = book_embedding_weights[book_id_encoded].reshape(1, -1)
    except ValueError:
        return {"error": f"Book '{book_title}' not found in the dataset."}

    similarities = cosine_similarity(book_vec, book_embedding_weights)[0]
    similar_book_indices = np.argsort(similarities)[::-1]

    recommendations = []
    rec_count = 0
    for idx in similar_book_indices:
        if idx == book_id_encoded:
            continue
        
        recommended_title = book_encoder.inverse_transform([idx])[0]
        similarity_score = float(similarities[idx])
        
        book_details = books_df[books_df['Book-Title'] == recommended_title].drop_duplicates('Book-Title')
        
        if not book_details.empty:
            rec_data = {
                "title": recommended_title,
                "author": book_details['Book-Author'].values[0],
                "year": str(book_details['Year-Of-Publication'].values[0]),
                "publisher": book_details['Publisher'].values[0],
                "image_url": book_details['Image-URL-M'].values[0],
                "similarity_score": similarity_score
            }
            recommendations.append(rec_data)
        
        rec_count += 1
        if rec_count >= num_recs:
            break
            
    return {"recommendations": recommendations}


# --- 3. The Flask API ---
app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>Book Recommendation API</h1><p>Use the /recommend endpoint, e.g., /recommend?title=The+Da+Vinci+Code</p>"

@app.route('/recommend', methods=['GET'])
def recommend():
    book_title = request.args.get('title')
    
    if not book_title:
        return jsonify({"error": "Please provide a 'title' query parameter."}), 400
        
    results = get_api_recommendations(book_title)
    
    return jsonify(results)

if __name__ == '__main__':
    # This is for local testing only
    app.run(debug=True)