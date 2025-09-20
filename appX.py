import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
import random # For dummy data generation

# --- Page Configuration ---
st.set_page_config(
    page_title="Personalized Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium UI ---
st.markdown("""
<style>
    /* ... (your existing Poppins font import and base styles) ... */

    /* Main Container - More subtle, clean background like YouTube */
    .stApp {
        background-color: #f9f9f9; /* Lighter background */
        min-height: 100vh;
    }
    
    /* Header - More YouTube-like */
    .app-header {
        background-color: #ffffff; /* White header */
        color: #000; /* Darker text */
        padding: 1rem 2rem; /* Reduced padding */
        border-bottom: 1px solid #e0e0e0; /* Subtle bottom border */
        box-shadow: 0 2px 8px rgba(0,0,0,0.05); /* Lighter shadow */
        margin-bottom: 1.5rem; /* Reduced margin */
        display: flex; /* Use flexbox for alignment */
        align-items: center;
        justify-content: space-between; /* Space out elements */
        border-radius: 0; /* No rounded corners at the bottom */
    }
    
    .app-header h1 {
        font-weight: 700;
        font-size: 1.8rem; /* Slightly smaller */
        margin-bottom: 0;
        color: #212121; /* Darker text */
    }
    
    .app-header h1 .icon {
        color: #ff0000; /* YouTube Red for the icon */
        margin-right: 0.5rem;
    }

    .app-header p {
        display: none; /* Hide subtitle for a cleaner look */
    }

    /* Search Bar in Header (Simulated, as Streamlit text_input is block-level) */
    /* This would need Streamlit components arranged in columns for true header integration */
    .stTextInput > div > div > input {
        border-radius: 20px; /* More rounded search bar */
        border: 1px solid #ccc;
        padding: 0.5rem 1rem;
        box-shadow: none;
    }

    /* Cards - Closer to YouTube video thumbnails */
    .custom-card, .movie-card {
        background: white;
        border-radius: 8px; /* Slightly less rounded */
        padding: 0; /* No padding directly on card */
        box-shadow: 0 2px 10px rgba(0,0,0,0.05); /* Lighter shadow */
        margin-bottom: 1.2rem; /* Reduced margin */
        overflow: hidden; /* Ensure content stays within borders */
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .custom-card:hover, .movie-card:hover {
        transform: translateY(-3px); /* Less dramatic lift */
        box-shadow: 0 8px 20px rgba(0,0,0,0.1); /* Slightly stronger hover shadow */
    }
    
    .card-header { /* For general info cards */
        padding: 1rem 1.5rem;
        margin-bottom: 0;
        border-bottom: 1px solid #f0f0f0;
    }

    .card-icon {
        background: #ff0000; /* YouTube red */
        border-radius: 50%; /* Make it round */
    }

    /* Movie Poster/Thumbnail - CRUCIAL FOR YOUTUBE LOOK */
    .movie-poster {
        height: 180px; /* Consistent height for thumbnails */
        width: 100%;
        background-color: #e2e8f0; /* Default placeholder background */
        display: flex;
        align-items: center;
        justify-content: center;
        color: #64748b;
        font-size: 3rem; /* Still shows first letter */
        border-bottom: none; /* No border */
        position: relative; /* For potential overlay info */
    }

    /* Placeholder text for missing posters */
    .movie-poster::after {
        content: attr(data-letter); /* Use data attribute for letter */
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 3em;
        color: #64748b;
    }
    .movie-poster img { /* If you add actual images */
        width: 100%;
        height: 100%;
        object-fit: cover; /* Cover the area, crop if necessary */
    }

    .movie-content {
        padding: 0.8rem 1rem; /* Smaller padding inside content */
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    
    .movie-title-rec {
        font-weight: 600;
        font-size: 1rem; /* Slightly smaller font */
        line-height: 1.3em; /* Tighten line height */
        margin-bottom: 0.3rem; /* Less space */
        min-height: 2.6em; /* Adjust for 2 lines of text */
        overflow: hidden; /* Hide overflow */
        text-overflow: ellipsis; /* Add ellipsis if text is too long */
        display: -webkit-box;
        -webkit-line-clamp: 2; /* Limit to 2 lines */
        -webkit-box-orient: vertical;
    }
    
    .movie-meta {
        font-size: 0.85rem; /* Smaller meta info */
        color: #606060; /* Darker grey like YouTube */
        margin-bottom: 0.4rem;
    }

    .movie-genres {
        display: none; /* Often hidden in main YouTube view to keep it clean */
    }
    
    .recommendation-info {
        font-size: 0.8em; /* Smaller info text */
        color: #747474; /* Lighter grey */
        margin-top: auto;
        padding-top: 0.4rem;
        border-top: 1px solid #f5f5f5; /* Lighter border */
    }
    
    /* Buttons - More subtle, less gradient */
    .stButton > button {
        background-color: #000000; /* Black button */
        color: white;
        border: none;
        padding: 0.6rem 1.2rem; /* Smaller padding */
        border-radius: 4px; /* Less rounded */
        font-weight: 500;
        transition: background-color 0.2s ease;
        width: 100%;
        margin-top: 0.5rem; /* Reduced margin */
    }
    
    .stButton > button:hover {
        background-color: #333333; /* Darker hover */
        transform: none; /* No translateY on hover */
        box-shadow: none; /* No shadow on hover */
    }

    /* Stars for Rating */
    .stButton > button[key*="rate_star_"] {
        background: none;
        border: none;
        color: #FFD700; /* Gold for stars */
        font-size: 1.5rem;
        padding: 0.2rem;
        box-shadow: none;
        width: auto;
        margin: 0;
    }
    .stButton > button[key*="rate_star_"]:hover {
        transform: scale(1.1);
        background: none;
        box-shadow: none;
    }

    /* Tabs - Clean, flat design */
    .stTabs [role="tablist"] button {
        background: none;
        border: none;
        border-bottom: 2px solid transparent; /* Highlight current tab with bottom border */
        border-radius: 0;
        padding: 0.7rem 1.2rem;
        font-weight: 500;
        transition: all 0.2s ease;
        color: #606060; /* Grey text */
        margin: 0 0.5rem;
    }
    
    .stTabs [role="tablist"] button:hover {
        color: #212121; /* Darker text on hover */
        background: none;
        transform: none;
        border-color: #e0e0e0; /* Subtle hover underline */
    }

    .stTabs [role="tab"][aria-selected="true"] {
        background: none;
        color: #212121; /* Darker selected text */
        border-bottom: 2px solid #ff0000; /* YouTube red underline for active tab */
        box-shadow: none;
        font-weight: 600;
    }

    /* Selectbox/Radio/etc. - Minimalist */
    .stSelectbox > div > div, .stRadio > label > div {
        border-radius: 4px;
        border: 1px solid #ccc;
        background-color: white;
        padding: 0.3rem;
    }

    /* Info/Warning messages */
    .stAlert {
        border-radius: 4px;
        padding: 1rem;
        font-size: 0.9em;
    }
    .stInfo { background-color: #e3f2fd; border-left: 5px solid #2196f3; color: #1e88e5; }
    .stWarning { background-color: #fffde7; border-left: 5px solid #ffc107; color: #ffa000; }
    .stSuccess { background-color: #e8f5e9; border-left: 5px solid #4caf50; color: #388e3c; }


    /* Footer - More subtle */
    div[data-testid="stStatusWidget"] {
        visibility: hidden; /* Hide the Streamlit footer */
    }
    .stApp footer {
        visibility: hidden; /* Hide the Streamlit footer */
    }

    /* Custom Footer styling */
    .app-footer {
        text-align: center;
        margin-top: 40px;
        padding: 15px;
        background: #ffffff;
        border-top: 1px solid #f0f0f0;
        color: #606060;
        font-size: 0.8em;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    """
    Loads movie and ratings data. If CSVs are not found, generates dummy data.
    Ensures all necessary columns for the recommendation system are present.
    """
    try:
        ratings = pd.read_csv("ratings.csv")
        movies = pd.read_csv("movies.csv")
        st.success("Loaded movies.csv and ratings.csv successfully!")
    except FileNotFoundError:
        st.warning("`movies.csv` or `ratings.csv` not found. Generating dummy data for demonstration.")
        
        # Generate dummy movie data with more realistic structure
        movie_data = {
            'movieId': range(1, 101), # 100 dummy movies
            'title': [
                f"{random.choice(['The', 'A', 'My'])} {random.choice(['Amazing', 'Dark', 'Forgotten', 'Silent', 'Last'])} {random.choice(['Journey', 'City', 'Star', 'Secret', 'Dream'])} {i}" 
                for i in range(1, 101)
            ],
            'genres': [
                "Action|Adventure|Sci-Fi", "Comedy|Drama|Romance", "Thriller|Mystery|Crime", "Drama|War", 
                "Fantasy|Animation|Children", "Horror", "Documentary", "Musical", "Western", "Family"
            ][np.random.randint(0, 10, 100)].tolist(),
            'year': np.random.randint(1980, 2024, 100).tolist(),
            'vote_average': np.round(np.random.uniform(5.0, 9.5, 100), 1).tolist(),
            'popularity': np.round(np.random.uniform(10.0, 200.0, 100), 2).tolist(),
            'description': [
                f"A compelling story about a {random.choice(['hero', 'villain', 'scientist', 'artist'])} who embarks on a {random.choice(['perilous', 'magical', 'dangerous', 'enlightening'])} journey to {random.choice(['save the world', 'discover himself', 'unravel a mystery', 'find true love'])}."
                for _ in range(100)
            ]
        }
        movies = pd.DataFrame(movie_data)

        # Generate dummy ratings data
        rating_data = []
        for user_id in range(1, 21): # 20 dummy users
            num_ratings = random.randint(10, 50) # Each user rates 10-50 movies
            rated_movies = random.sample(movies['movieId'].tolist(), num_ratings) # Ensure unique movie IDs
            for movie_id in rated_movies:
                rating = random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
                # Add dummy timestamp
                timestamp = int(time.time()) - random.randint(0, 365 * 24 * 60 * 60) # Last year
                rating_data.append({'userId': user_id, 'movieId': movie_id, 'rating': rating, 'timestamp': timestamp})
        ratings = pd.DataFrame(rating_data)
        st.info("Using generated dummy data. For real experience, please provide `movies.csv` and `ratings.csv`.")

    # Ensure necessary columns are present and handle NaNs
    if 'genres' not in movies.columns: movies['genres'] = ''
    if 'description' not in movies.columns: movies['description'] = movies['title'].apply(lambda x: f"A film about {x}")
    if 'year' not in movies.columns: movies['year'] = 2000
    if 'vote_average' not in movies.columns: movies['vote_average'] = 7.0
    if 'popularity' not in movies.columns: movies['popularity'] = 50.0

    df = pd.merge(ratings, movies, on="movieId")
    df.dropna(subset=['rating', 'genres', 'description'], inplace=True) # Drop rows with essential missing data

    # Normalize ratings (if needed for certain models, e.g., for direct comparison across users)
    scaler = MinMaxScaler(feature_range=(1, 5)) # Scale ratings to 1-5 range for consistency if not already
    df['norm_rating'] = df['rating'] # Using original rating as normalized for simplicity unless specific model needs 0-1
    # df['norm_rating'] = scaler.fit_transform(df[['rating']])


    return df, movies

# Load data and get the main DataFrame
df_data, movies_df = load_and_preprocess_data()

# --- Session State Initialization ---
if 'user_session_ratings' not in st.session_state:
    st.session_state.user_session_ratings = {}
if 'recommendations_triggered' not in st.session_state:
    st.session_state.recommendations_triggered = False

# --- Model Initialization ---
@st.cache_resource
def initialize_models(df_processed, movies_data):
    """
    Initializes and trains all recommendation models.
    Caches the results to avoid recomputation on every rerun.
    """
    st.info("Initializing recommendation models. This may take a moment...")

    # Collaborative Filtering setup (User-Item Matrix)
    user_item_matrix = None
    user_similarity_matrix = None
    item_similarity_matrix = None
    svd_model = None
    svd_matrix = None
    # nmf_model = None # NMF not used in current version, commented out to avoid warnings if not needed.
    # nmf_matrix = None

    if not df_processed.empty and all(col in df_processed.columns for col in ['userId', 'movieId', 'rating']):
        try:
            user_item_matrix = df_processed.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
            
            # User-User Similarity
            user_similarity_matrix = cosine_similarity(user_item_matrix)
            user_similarity_df = pd.DataFrame(user_similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

            # Item-Item Similarity
            item_similarity_matrix = cosine_similarity(user_item_matrix.T)
            item_similarity_df = pd.DataFrame(item_similarity_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)

            # Matrix Factorization (SVD)
            # Adjust n_components based on dataset size and computational resources
            n_components_svd = min(20, user_item_matrix.shape[0]-1, user_item_matrix.shape[1]-1)
            if n_components_svd > 0:
                svd_model = TruncatedSVD(n_components=n_components_svd, random_state=42)
                svd_matrix = svd_model.fit_transform(user_item_matrix)
                # Reconstruct matrix for RMSE/MAE calculation
                svd_predicted_ratings = np.dot(svd_matrix, svd_model.components_)
                # Ensure comparison only on non-zero original values
                true_ratings_flat = user_item_matrix.values[user_item_matrix.values != 0]
                pred_ratings_flat = svd_predicted_ratings[user_item_matrix.values != 0]
                
                svd_rmse = np.sqrt(mean_squared_error(true_ratings_flat, pred_ratings_flat))
                svd_mae = mean_absolute_error(true_ratings_flat, pred_ratings_flat)
                st.session_state['svd_rmse'] = svd_rmse
                st.session_state['svd_mae'] = svd_mae
            else:
                st.warning("Not enough data for SVD.")
            
            st.success("Collaborative Filtering models initialized!")

        except Exception as e:
            st.error(f"Error initializing Collaborative Filtering models: {e}")
            user_item_matrix = None
            user_similarity_matrix = None
            item_similarity_matrix = None
            svd_model = None
            svd_matrix = None
    else:
        st.warning("Insufficient data for Collaborative Filtering. Skipping initialization.")


    # Content-Based Filtering setup (TF-IDF)
    tfidf_vectorizer = None
    tfidf_matrix = None
    genre_similarity_matrix = None

    # Use 'description' if available and not empty, otherwise 'genres'
    content_source = movies_data['description'] if 'description' in movies_data.columns and not movies_data['description'].empty else movies_data['genres'].fillna('')
    
    if not content_source.empty:
        try:
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf_vectorizer.fit_transform(content_source)
            genre_similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix) # Using linear_kernel for cosine similarity
            genre_similarity_df = pd.DataFrame(genre_similarity_matrix, index=movies_data['movieId'], columns=movies_data['movieId'])
            st.success("Content-Based Filtering model initialized!")
        except Exception as e:
            st.error(f"Error initializing Content-Based Filtering model: {e}")
            tfidf_vectorizer = None
            tfidf_matrix = None
            genre_similarity_matrix = None
    else:
        st.warning("Insufficient content data for Content-Based Filtering. Skipping initialization.")
    
    st.info("All models initialized!")
    return {
        "user_item_matrix": user_item_matrix,
        "user_similarity_df": user_similarity_df if 'user_similarity_df' in locals() else None,
        "item_similarity_df": item_similarity_df if 'item_similarity_df' in locals() else None,
        "svd_model": svd_model,
        "svd_matrix": svd_matrix,
        # "nmf_model": nmf_model,
        # "nmf_matrix": nmf_matrix,
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "genre_similarity_df": genre_similarity_df if 'genre_similarity_df' in locals() else None
    }

models = initialize_models(df_data, movies_df)

# --- Recommendation Functions ---

def get_collaborative_recommendations(user_id, n=5, session_ratings=None):
    """
    Generates collaborative filtering recommendations.
    Prioritizes session_ratings if provided, otherwise uses historical user_id data.
    """
    user_item_m = models.get("user_item_matrix")
    item_sim_df = models.get("item_similarity_df")
    
    if user_item_m is None or item_sim_df is None:
        return pd.DataFrame(), "Collaborative model not ready."

    current_user_ratings = {}
    is_session_user = False

    if session_ratings:
        current_user_ratings = session_ratings
        is_session_user = True
    elif user_id in user_item_m.index:
        current_user_ratings = user_item_m.loc[user_id].dropna().to_dict()
    
    if not current_user_ratings:
        return movies_df.sort_values('popularity', ascending=False).head(n).copy().assign(reason="Popular movies (no user ratings for CF)", predicted_rating=3.5), "No ratings for CF model."

    try:
        # Create a temporary user-item series for the current user's ratings
        temp_user_series = pd.Series(current_user_ratings)
        
        # Identify movies already rated by the current user (from both session and historical if applicable)
        user_rated_movies = set(current_user_ratings.keys())
        
        # Get all movies available in the item similarity matrix
        all_movies_in_sim = item_sim_df.index
        
        # Calculate predicted ratings for all unrated movies
        unrated_movies = all_movies_in_sim.difference(user_rated_movies)
        predicted_ratings = pd.Series(index=unrated_movies, dtype=float)

        for movie_id in unrated_movies:
            # Get similarity scores for this unrated movie with movies rated by the current user
            if movie_id in item_sim_df.index: # Ensure target movie exists in similarity matrix
                # Get similarities of 'movie_id' with all movies current_user_ratings
                sim_scores_with_rated = item_sim_df.loc[movie_id, temp_user_series.index]
                
                # Filter out NaN similarities and common movies not in item_sim_df
                sim_scores_with_rated = sim_scores_with_rated.dropna() 
                
                if not sim_scores_with_rated.empty and sim_scores_with_rated.sum() > 0:
                    # Calculate weighted sum of ratings from current user's rated movies
                    weighted_sum = (sim_scores_with_rated * temp_user_series[sim_scores_with_rated.index]).sum()
                    sum_of_weights = sim_scores_with_rated.sum()
                    predicted_ratings.loc[movie_id] = weighted_sum / sum_of_weights
                else:
                    predicted_ratings.loc[movie_id] = 0 # No sufficient similar items rated by user

        predicted_ratings = predicted_ratings.dropna().sort_values(ascending=False)
        
        recs_df = movies_df[movies_df['movieId'].isin(predicted_ratings.head(n).index)].copy()
        recs_df['predicted_rating'] = recs_df['movieId'].map(predicted_ratings)
        recs_df['reason'] = "Based on items similar to your liked movies"
        return recs_df, "Collaborative Filtering"

    except Exception as e:
        return pd.DataFrame(), f"Error in Collaborative Filtering: {e}"

def get_content_based_recommendations(user_id, n=5, session_ratings=None, movie_id_seed=None):
    """
    Generates content-based recommendations.
    Prioritizes building a user profile from session_ratings, otherwise uses movie_id_seed.
    """
    tfidf_v = models.get("tfidf_vectorizer")
    tfidf_m = models.get("tfidf_matrix")
    
    if tfidf_v is None or tfidf_m is None:
        return pd.DataFrame(), "Content-based model not ready."

    rated_movies_for_profile = {}
    if session_ratings:
        rated_movies_for_profile = session_ratings
    elif user_id in df_data['userId'].unique(): # Check if user has historical ratings
        user_historical_ratings = df_data[df_data['userId'] == user_id][['movieId', 'rating']].set_index('movieId')['rating'].to_dict()
        rated_movies_for_profile = user_historical_ratings

    user_profile_vector = None
    if rated_movies_for_profile:
        # Build user profile from content of rated movies
        profile_vectors = []
        for mid, rating in rated_movies_for_profile.items():
            movie_idx = movies_df[movies_df['movieId'] == mid].index
            if not movie_idx.empty:
                # Weight by rating (squared for stronger preference)
                profile_vectors.append(tfidf_m[movie_idx[0]] * (rating ** 2))
        
        if profile_vectors:
            # Sum the sparse vectors. The result will still be a sparse matrix (1, num_features)
            user_profile_vector_sparse = np.sum(profile_vectors, axis=0)
            
            # Convert to a dense 1D numpy array for norm calculation
            user_profile_vector = user_profile_vector_sparse.toarray().flatten()

            norm_val = np.linalg.norm(user_profile_vector)
            if norm_val > 0:
                user_profile_vector = user_profile_vector / norm_val # Normalize
            else:
                user_profile_vector = None # Cannot normalize a zero vector
                # No need to set reason_prefix here, it's handled by the outer if user_profile_vector is None
        
        reason_prefix = "Based on your preferred genres and content"

    elif movie_id_seed is not None:
        # Fallback to single movie similarity if no user ratings or session ratings
        idx = movies_df[movies_df['movieId'] == movie_id_seed].index[0]
        user_profile_vector = tfidf_m[idx].toarray().flatten() # Ensure dense for consistency
        reason_prefix = f"Similar to '{movies_df.loc[idx, 'title']}'"

    else:
        return pd.DataFrame(), "No basis for content-based recommendations (no user ratings or seed movie)."

    if user_profile_vector is None or np.all(user_profile_vector == 0): # Check for all zeros after conversion
        return pd.DataFrame(), "Could not create a valid content-based user profile."

    try:
        cosine_sim = linear_kernel(user_profile_vector.reshape(1, -1), tfidf_m).flatten() # Reshape for linear_kernel

        # Get top N most similar movies, excluding movies already rated by the current user
        # (This combines session_ratings and existing df_data for exclusion)
        all_rated_by_current_user = set(st.session_state.user_session_ratings.keys())
        if user_id in df_data['userId'].unique():
            all_rated_by_current_user.update(df_data[df_data['userId'] == user_id]['movieId'].unique())

        sim_scores_with_ids = sorted(
            [(movies_df.iloc[i]['movieId'], score) for i, score in enumerate(cosine_sim) 
             if movies_df.iloc[i]['movieId'] not in all_rated_by_current_user],
            key=lambda x: x[1], reverse=True
        )
        
        top_n_sim_movies = sim_scores_with_ids[:n]
        recs_df = movies_df[movies_df['movieId'].isin([mid for mid, _ in top_n_sim_movies])].copy()
        recs_df['predicted_rating'] = [score * 5 for _, score in top_n_sim_movies] # Scale similarity to 0-5
        recs_df['reason'] = reason_prefix + " (content-wise)"

        return recs_df, "Content-Based Filtering"

    except Exception as e:
        return pd.DataFrame(), f"Error in Content-Based Filtering: {e}"


def get_hybrid_recommendations(user_id, n=5, session_ratings=None, movie_id_seed=None, col_weight=0.6, content_weight=0.4):
    """
    Generates hybrid recommendations by combining collaborative and content-based scores.
    """
    # Call collaborative and content-based functions, passing session_ratings if available
    collab_recs_df, collab_reason_msg = get_collaborative_recommendations(user_id, n=n*3, session_ratings=session_ratings) 
    content_recs_df, content_reason_msg = get_content_based_recommendations(user_id, n=n*3, session_ratings=session_ratings, movie_id_seed=movie_id_seed)

    combined_recs = {}

    # Add collaborative recommendations to combined_recs
    if not collab_recs_df.empty:
        for _, row in collab_recs_df.iterrows():
            movie_id = row['movieId']
            combined_recs[movie_id] = {
                'movie_info': row.to_dict(),
                'collab_score': row['predicted_rating'],
                'content_score': 0.0, # Initialize
                'reason_collab': row['reason']
            }

    # Add content-based recommendations, updating if movie already exists
    if not content_recs_df.empty:
        for _, row in content_recs_df.iterrows():
            movie_id = row['movieId']
            if movie_id in combined_recs:
                combined_recs[movie_id]['content_score'] = row['predicted_rating']
                combined_recs[movie_id]['reason_content'] = row['reason']
            else:
                combined_recs[movie_id] = {
                    'movie_info': row.to_dict(),
                    'collab_score': 0.0, # Initialize
                    'content_score': row['predicted_rating'],
                    'reason_content': row['reason']
                }
    
    final_recs_list = []
    for movie_id, data in combined_recs.items():
        movie = data['movie_info']
        collab_score = data['collab_score']
        content_score = data['content_score']

        # Ensure that if one score is 0 (meaning no recommendation from that method), it doesn't skew the average
        effective_col_weight = col_weight if collab_score > 0 else 0
        effective_content_weight = content_weight if content_score > 0 else 0
        
        total_effective_weight = effective_col_weight + effective_content_weight

        if total_effective_weight > 0:
            hybrid_score = (collab_score * effective_col_weight + content_score * effective_content_weight) / total_effective_weight
        elif collab_score > 0: # Fallback if only collaborative had score
            hybrid_score = collab_score
        elif content_score > 0: # Fallback if only content had score
            hybrid_score = content_score
        else:
            hybrid_score = 0 # This case implies neither method gave a score, should ideally not be in combined_recs.items()

        # Build reason string
        reasons = []
        if 'reason_collab' in data and data['reason_collab']:
            reasons.append(data['reason_collab'])
        if 'reason_content' in data and data['reason_content']:
            reasons.append(data['reason_content'])
        
        reason_str = " & ".join(reasons) if reasons else "General recommendation"

        final_recs_list.append({
            **movie, # Include all movie info
            'predicted_rating': hybrid_score,
            'reason': reason_str
        })

    final_recs_df = pd.DataFrame(final_recs_list)

    if not final_recs_df.empty:
        # Filter out movies already rated by the user (from session state)
        user_seen_movies_session = set(st.session_state.user_session_ratings.keys())
        final_recs_df = final_recs_df[~final_recs_df['movieId'].isin(user_seen_movies_session)]
        
        # Also filter out movies rated by the historically selected user if applicable
        if user_id in df_data['userId'].unique():
             user_seen_movies_historical = df_data[df_data['userId'] == user_id]['movieId'].unique()
             final_recs_df = final_recs_df[~final_recs_df['movieId'].isin(user_seen_movies_historical)]


        final_recs_df = final_recs_df.sort_values('predicted_rating', ascending=False).drop_duplicates(subset=['movieId']).head(n)
    
    # Fallback if no personalized recommendations are found
    if final_recs_df.empty:
        final_recs_df = movies_df.sort_values('popularity', ascending=False).head(n).copy()
        final_recs_df['predicted_rating'] = 3.5 # Default rating for popular fallback
        final_recs_df['reason'] = "Popular movies (default fallback, no personalized recommendations found)"

    return final_recs_df, "Hybrid Filtering"


# --- UI Components ---
def display_movie_card(movie_data, show_predicted_rating=False, reason_text=None, is_rateable=False):
    """
    Renders a single movie card with dynamic content and optional rating/info.
    `is_rateable` makes the stars clickable to rate.
    """
    
    # Generate an image placeholder using the first letter of the title
    first_letter = movie_data['title'][0].upper() if pd.notna(movie_data['title']) and movie_data['title'] else '?'
    
    card_html = f"""
    <div class="movie-card" style="min-height: 380px;">
        <div class="movie-poster" data-letter="{first_letter}">
        </div>
        <div class="movie-content">
            <div class="movie-title-rec">{movie_data['title']}</div>
            <div class="movie-meta">
                <span>{movie_data['year']}</span>
                <span>⭐ {movie_data['vote_average']:.1f}</span>
            </div>
            <div class="movie-genres">
                {''.join([f'<span class="genre-tag">{genre.strip()}</span>' for genre in movie_data['genres'].split('|')[:3]])}
            </div>
            """
    
    if show_predicted_rating:
        card_html += f"""<p class="recommendation-info">Predicted: <b>{movie_data['predicted_rating'] if 'predicted_rating' in movie_data else 'N/A':.1f}</b> ⭐</p>"""
    if reason_text:
        card_html += f"""<p class="recommendation-info"><i>{reason_text}</i></p>"""

    card_html += """
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    
    if is_rateable:
        current_rating = st.session_state.user_session_ratings.get(movie_data['movieId'], 0)
        
        cols = st.columns(5) # 5 columns for 5 stars
        for k in range(1, 6):
            with cols[k-1]:
                # Use a unique key for each star button
                star_key = f"rate_star_{movie_data['movieId']}_{k}"
                if st.button("⭐", key=star_key, help=f"Rate {k} stars"):
                    st.session_state.user_session_ratings[movie_data['movieId']] = k
                    st.toast(f"Rated '{movie_data['title']}' {k} stars!")
                    if len(st.session_state.user_session_ratings) >= 3 and not st.session_state.recommendations_triggered:
                        st.session_state.recommendations_triggered = True
                        st.toast("Great! You've rated 3 movies. Check 'Get Your Recommendations' tab!", icon="🎉")
                        time.sleep(0.5) # Short delay to allow toast to show
                    st.rerun() # Rerun to update the display

        if current_rating > 0:
            st.markdown(f'<p style="text-align: center; font-size: 0.9em; font-weight: bold; color: #4f46e5;">Your Rating: {current_rating} ⭐</p>', unsafe_allow_html=True)


# --- Main Application ---
st.markdown("""
<div class="app-header">
    <h1><span class="icon">▶️</span> CineMatch AI</h1>
    </div>
""", unsafe_allow_html=True)

# Sidebar for global controls (like user selection)
user_ids = sorted(df_data['userId'].unique().tolist())
selected_user = st.sidebar.selectbox("Select Historical User ID (for Insights tabs)", user_ids, key="sidebar_user_select")
st.sidebar.markdown("---")
st.sidebar.info("Select a user to explore pre-calculated model insights. For *interactive* recommendations, use the 'Rate Movies' tab.")

# Tabs for different phases
tab_titles = ["📊 EDA", "👥 Collaborative Filtering Insights", "🧠 Content-Based Filtering Insights", "🔀 Hybrid Model Insights", "🌟 Rate Movies", "🎯 Get Your Recommendations"]
tabs = st.tabs(tab_titles)

# --- EDA Tab ---
with tabs[0]:
    st.header("📊 Exploratory Data Analysis")
    st.markdown("Dive into the raw data to understand its distribution and key characteristics.")

    col_stats1, col_stats2, col_stats3 = st.columns(3)
    with col_stats1:
        st.metric("Total Movies", f"{movies_df['movieId'].nunique():,}")
    with col_stats2:
        st.metric("Total Users", f"{df_data['userId'].nunique():,}")
    with col_stats3:
        st.metric("Total Ratings", f"{len(df_data):,}")

    st.subheader("Rating Distribution")
    fig_ratings_dist = px.histogram(df_data, x="rating", nbins=10, 
                                    title="Distribution of Movie Ratings (1-5 Stars)",
                                    color_discrete_sequence=px.colors.sequential.Viridis)
    fig_ratings_dist.update_layout(xaxis_title="Rating (Stars)", yaxis_title="Number of Ratings")
    st.plotly_chart(fig_ratings_dist, use_container_width=True)

    st.subheader("Top 10 Most Rated Movies")
    top_rated_movies = df_data.groupby("title")['rating'].mean().sort_values(ascending=False).head(10).reset_index()
    top_rated_movies.columns = ['Movie Title', 'Average Rating']
    st.dataframe(top_rated_movies, use_container_width=True, hide_index=True)

    st.subheader("Genre Distribution")
    # Explode genres into separate rows for counting
    all_genres = df_data['genres'].str.split('|', expand=True).stack().reset_index(level=1, drop=True)
    genre_counts = all_genres.value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Count']
    fig_genres_dist = px.bar(genre_counts.head(15), x='Count', y='Genre', orientation='h',
                             title="Top 15 Most Common Genres in Dataset",
                             color_discrete_sequence=px.colors.sequential.Plasma)
    fig_genres_dist.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort bars
    st.plotly_chart(fig_genres_dist, use_container_width=True)


# --- Collaborative Filtering Tab ---
with tabs[1]:
    st.header("👥 Collaborative Filtering Insights")
    st.markdown("""
    This approach recommends movies based on the preferences of users with similar tastes. 
    "Users who liked what you liked, also liked..."
    """)

    if models.get("user_item_matrix") is not None:
        st.subheader("How it works:")
        st.markdown("""
        1.  **User-Item Matrix:** A matrix is created where rows are users, columns are movies, and cell values are ratings. Missing values (unrated movies) are typically filled with zeros or user averages.
        2.  **Similarity Calculation:** We find users (User-User CF) or items (Item-Item CF) that are most similar to the target user/item based on their rating patterns. Cosine similarity is a common metric.
        3.  **Prediction & Recommendation:** For unrated movies, a predicted rating is calculated based on the ratings of similar users (or ratings of similar items). The highest-predicted unrated movies are recommended.
        """)
        
        st.subheader("Key Components & Metrics:")
        col_cf1, col_cf2 = st.columns(2)
        with col_cf1:
            st.metric("User-Item Matrix Size", f"{models['user_item_matrix'].shape[0]} Users x {models['user_item_matrix'].shape[1]} Movies")
            # Corrected formatting for svd_rmse and svd_mae
            st.metric("SVD RMSE", f"{st.session_state.get('svd_rmse', 'N/A'):.4f}" if isinstance(st.session_state.get('svd_rmse'), (int, float)) else f"{st.session_state.get('svd_rmse', 'N/A')}", help="Root Mean Squared Error for SVD predictions.")
            st.metric("SVD MAE", f"{st.session_state.get('svd_mae', 'N/A'):.4f}" if isinstance(st.session_state.get('svd_mae'), (int, float)) else f"{st.session_state.get('svd_mae', 'N/A')}", help="Mean Absolute Error for SVD predictions.")
        with col_cf2:
            st.metric("Personalization", "High", help="CF models excel at tailored recommendations.")
            st.metric("Cold Start Problem", "Significant", help="Struggles with new users/items without sufficient ratings.")
        
        st.markdown("#### User-User Similarity (Sample Heatmap)")
        if models.get("user_similarity_df") is not None and not models["user_similarity_df"].empty:
            # Select a random subset of users for heatmap for better visualization
            sample_users_for_heatmap = random.sample(user_ids, min(10, len(user_ids)))
            sample_sim_df = models["user_similarity_df"].loc[sample_users_for_heatmap, sample_users_for_heatmap]
            
            fig, ax = plt.subplots(figsize=(8, 7))
            sns.heatmap(sample_sim_df, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, ax=ax)
            ax.set_title(f"Cosine Similarity between Sample Users")
            st.pyplot(fig)
            st.info("Higher values indicate more similar rating patterns between users.")
        else:
            st.info("User similarity matrix not available or empty for visualization.")

        st.markdown("#### Example: Top Movies Rated by Similar Users to Historical User " + str(selected_user))
        
        # Get actual top recommendations for selected user using CF
        cf_recs, cf_msg = get_collaborative_recommendations(selected_user, n=5)
        if not cf_recs.empty:
            cols_cf = st.columns(5)
            for idx, movie in cf_recs.iterrows():
                with cols_cf[idx % 5]:
                    display_movie_card(movie, show_predicted_rating=True, reason_text=movie.get('reason', ''))
        else:
            st.info("No collaborative recommendations available for the selected historical user. " + cf_msg)

    else:
        st.warning("Collaborative Filtering models could not be initialized due to missing or insufficient data.")


# --- Content-Based Filtering Tab ---
with tabs[2]:
    st.header("🧠 Content-Based Filtering Insights")
    st.markdown("""
    This approach recommends movies based on their features (content) like genres and descriptions.
    "If you liked this movie, you might like others with similar characteristics."
    """)

    if models.get("tfidf_vectorizer") is not None:
        st.subheader("How it works:")
        st.markdown("""
        1.  **Text Vectorization:** Movie descriptions and genres are converted into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency). This identifies important keywords for each movie.
        2.  **Content Similarity:** Cosine similarity is calculated between a target movie's vector and all other movie vectors.
        3.  **Recommendation:** Movies with the highest content similarity scores are recommended.
        """)

        st.subheader("Key Components & Metrics:")
        col_cb1, col_cb2 = st.columns(2)
        with col_cb1:
            st.metric("TF-IDF Features", f"{models['tfidf_matrix'].shape[1]:,} unique terms")
            st.metric("Sparsity", f"{100 * (1 - models['tfidf_matrix'].nnz / (models['tfidf_matrix'].shape[0] * models['tfidf_matrix'].shape[1])):.2f}%", help="Percentage of zero values in the TF-IDF matrix.")
        with col_cb2:
            st.metric("Diversity", "High", help="Can recommend movies across different user communities, only relying on content.")
            st.metric("Cold Start Problem (New Items)", "Low Impact", help="Can recommend new movies immediately based on their content.")
        
        st.markdown("#### Find Movies Similar to a Chosen Movie (Content-Wise)")
        selected_movie_for_content = st.selectbox(
            "Select a movie to find similar ones:", 
            movies_df['title'].values, 
            key="content_movie_select"
        )
        
        # Get the movieId from the title
        selected_movie_id = movies_df[movies_df['title'] == selected_movie_for_content]['movieId'].iloc[0]

        cb_recs, cb_msg = get_content_based_recommendations(selected_user, n=5, movie_id_seed=selected_movie_id) # Pass selected_user and movie_id_seed
        if not cb_recs.empty:
            cols_cb = st.columns(5)
            for idx, movie in cb_recs.iterrows():
                with cols_cb[idx % 5]:
                    display_movie_card(movie, show_predicted_rating=True, reason_text=movie.get('reason', ''))
        else:
            st.info("No content-based recommendations found for the selected movie. " + cb_msg)
    else:
        st.warning("Content-Based Filtering model could not be initialized due to missing or insufficient content data.")


# --- Hybrid Model Tab ---
with tabs[3]:
    st.header("🔀 Hybrid Recommendation System Insights")
    st.markdown("""
    The Hybrid approach combines the strengths of both Collaborative Filtering and Content-Based Filtering.
    This often leads to more robust, accurate, and diverse recommendations.
    """)

    st.subheader("How it works:")
    st.markdown("""
    1.  **Dual Recommendation:** Recommendations are generated independently using both collaborative and content-based methods.
    2.  **Score Combination:** The predicted ratings or similarity scores from both methods are combined (e.g., using a weighted average).
    3.  **Refined Recommendation:** The final recommendations are based on these combined scores, often prioritizing unrated movies.
    """)
    st.info("This system currently uses a weighted average of Collaborative (0.6) and Content-Based (0.4) predicted scores.")

    st.subheader("Key Advantages:")
    col_hyb1, col_hyb2, col_hyb3 = st.columns(3)
    with col_hyb1:
        st.metric("Addresses Cold Start", "Yes", help="Can recommend to new users/items using content if collaborative data is scarce.")
    with col_hyb2:
        st.metric("Mitigates Sparsity", "Yes", help="Can make recommendations even when similar users haven't rated an item.")
    with col_hyb3:
        st.metric("Improved Diversity", "Yes", help="Combines different recommendation strategies, leading to a broader range of suggestions.")

    st.markdown("#### Conceptual Model Comparison")
    
    comparison_data = {
        "Model": ["Content-Based", "Collaborative", "Hybrid"],
        "Diversity": [0.75, 0.65, 0.85], # Example values (conceptual)
        "Coverage": [0.90, 0.80, 0.95], # Example values (conceptual)
        "Personalization": [0.60, 0.85, 0.90] # Example values (conceptual)
    }
    comparison_df = pd.DataFrame(comparison_data)

    st.dataframe(comparison_df.set_index('Model'), use_container_width=True)
    
    fig = px.bar(
        comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
        x="Model",
        y="Score",
        color="Metric",
        title="Conceptual Model Comparison Metrics",
        barmode="group",
        height=400,
        color_discrete_map={
            "Diversity": "#4f46e5",
            "Coverage": "#7c3aed",
            "Personalization": "#10b981"
        }
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Rate Movies Tab ---
with tabs[4]:
    st.header("🌟 Rate Movies & Personalize")
    st.markdown("Help CineMatch AI learn your tastes! Rate at least **3 movies** to unlock personalized recommendations.")
    
    if len(st.session_state.user_session_ratings) < 3:
        st.info(f"You have rated **{len(st.session_state.user_session_ratings)}** movies this session. Rate **{3 - len(st.session_state.user_session_ratings)} more** to get started!")
    else:
        st.success(f"Great! You've rated {len(st.session_state.user_session_ratings)} movies. Now head to the '🎯 Get Your Recommendations' tab!")

    # Search and filter for movies to rate
    search_query_rate = st.text_input("Search movie title to rate:", key="search_rate_input_tab")
    
    all_genres = sorted(list(set(g for genres_str in movies_df['genres'].dropna() for g in genres_str.split('|'))))
    selected_genres_rate = st.multiselect("Filter by Genre:", all_genres, key="genre_filter_rate_tab")

    # Filter movies that haven't been rated in the current session
    unrated_movies = movies_df[~movies_df['movieId'].isin(st.session_state.user_session_ratings.keys())]

    filtered_movies_to_rate = unrated_movies[
        (unrated_movies['title'].str.contains(search_query_rate, case=False, na=False)) &
        (unrated_movies['genres'].apply(lambda x: any(g in x.split('|') for g in selected_genres_rate) if selected_genres_rate else True))
    ].sort_values('popularity', ascending=False) # Show popular unrated movies first

    if not filtered_movies_to_rate.empty:
        st.write("### Movies to Discover:")
        cols_per_row_rate = 3
        for i in range(0, len(filtered_movies_to_rate), cols_per_row_rate):
            cols = st.columns(cols_per_row_rate)
            for j in range(cols_per_row_rate):
                if i + j < len(filtered_movies_to_rate):
                    movie = filtered_movies_to_rate.iloc[i+j].to_dict()
                    with cols[j]:
                        display_movie_card(movie, is_rateable=True)
    else:
        st.info("No unrated movies found matching your search/filter. Try adjusting criteria or you've rated them all!")

    st.subheader("Your Rated Movies (This Session)")
    if st.session_state.user_session_ratings:
        session_rated_movies = movies_df[movies_df['movieId'].isin(st.session_state.user_session_ratings.keys())].copy()
        session_rated_movies['user_rating'] = session_rated_movies['movieId'].map(st.session_state.user_session_ratings)
        session_rated_movies = session_rated_movies.sort_values('user_rating', ascending=False)

        cols_per_row_session_rated = 3
        for i in range(0, len(session_rated_movies), cols_per_row_session_rated):
            cols = st.columns(cols_per_row_session_rated)
            for j in range(cols_per_row_session_rated):
                if i + j < len(session_rated_movies):
                    movie = session_rated_movies.iloc[i+j].to_dict()
                    with cols[j]:
                        st.markdown(f'<div class="movie-card" style="min-height: 380px;">', unsafe_allow_html=True)
                        st.markdown(f'<div class="movie-poster">{movie["title"][0]}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="movie-content" style="padding_bottom: 0.5rem;">', unsafe_allow_html=True)
                        st.markdown(f'<div class="movie-title-rec" style="min_height: 2.5em;">{movie["title"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="movie-meta"><span>{movie["year"]}</span><span>⭐ {movie["vote_average"]:.1f}</span></div>', unsafe_allow_html=True)
                        st.markdown('<div class="movie-genres">', unsafe_allow_html=True)
                        for genre in movie['genres'].split('|')[:3]:
                            st.markdown(f'<span class="genre-tag">{genre}</span>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown(f'<p style="text-align: center; font_size: 0.9em; font_weight: bold; color: #4f46e5; margin_top: auto; padding_top: 0.5rem; border_top: 1px solid #f0f0f0;">Your Rating: {movie["user_rating"]} ⭐</p>', unsafe_allow_html=True)
                        if st.button("Remove", key=f"remove_session_rating_{movie['movieId']}", use_container_width=True):
                            del st.session_state.user_session_ratings[movie['movieId']]
                            st.toast(f"Removed rating for '{movie['title']}'!")
                            st.session_state.recommendations_triggered = False # Reset trigger if ratings drop below 3
                            time.sleep(0.5)
                            st.rerun()
                        st.markdown('</div></div>', unsafe_allow_html=True) # Close movie-content and movie-card
    else:
        st.info("No movies rated in this session yet. Start rating above!")


# --- Get Your Recommendations Tab (Main Interaction) ---
with tabs[5]:
    st.header("🎯 Get Your Personalized Recommendations")
    st.markdown("Your recommendations are based on the movies you've rated in this session. The more you rate, the better the suggestions!")

    if len(st.session_state.user_session_ratings) < 3:
        st.warning("Please rate at least **3 movies** in the '🌟 Rate Movies' tab to get personalized recommendations.")
    else:
        st.info(f"Using your **{len(st.session_state.user_session_ratings)}** rated movies for personalization!")

        # Model Selection Radio Buttons
        selected_rec_model = st.radio(
            "Choose Recommendation Model:",
            ["Hybrid (Recommended)", "Collaborative Filtering", "Content-Based"],
            key="main_rec_model_select",
            horizontal=True,
            disabled=(len(st.session_state.user_session_ratings) < 1) # Disable if no ratings
        )

        # Seed movie for Content-Based/Hybrid if needed (can be based on user's highest rated)
        selected_seed_movie_id = None
        if st.session_state.user_session_ratings:
            # Find the highest rated movie in session to use as a seed, or a random one
            top_rated_session_movie_id = max(st.session_state.user_session_ratings, key=st.session_state.user_session_ratings.get)
            selected_seed_movie_title = movies_df[movies_df['movieId'] == top_rated_session_movie_id]['title'].iloc[0]
            selected_seed_movie_id = top_rated_session_movie_id
            
            st.markdown(f"**Content-Based/Hybrid Seed Movie:** Using '{selected_seed_movie_title}' (your highest rated movie in this session) as a starting point for content analysis.")
        else:
            st.warning("Rate movies in the 'Rate Movies' tab to enable personalized content-based recommendations.")
            # Fallback for seed if no session ratings
            selected_seed_movie_id = movies_df.sort_values('popularity', ascending=False)['movieId'].iloc[0]

        if st.button("Generate My Recommendations", key="generate_recs_button", use_container_width=True, disabled=(len(st.session_state.user_session_ratings) < 1)):
            st.subheader("Your Top Recommendations:")
            with st.spinner("Crunching data for your personalized movie suggestions..."):
                time.sleep(1.5) # Simulate processing

                final_recommendations_df = pd.DataFrame()
                rec_method_used = ""

                # Pass user_id (for historical context if needed by CF), but prioritize session_ratings
                if selected_rec_model == "Collaborative Filtering":
                    final_recommendations_df, rec_method_used = get_collaborative_recommendations(selected_user, n=10, session_ratings=st.session_state.user_session_ratings)
                elif selected_rec_model == "Content-Based":
                    if selected_seed_movie_id:
                        # For content-based, we want a user profile from *all* their rated movies, not just a single seed.
                        # The function get_content_based_recommendations already builds a profile from session_ratings.
                        final_recommendations_df, rec_method_used = get_content_based_recommendations(selected_user, n=10, session_ratings=st.session_state.user_session_ratings)
                    else:
                        st.error("Please rate movies to enable content-based recommendations.")
                else: # Hybrid (Recommended)
                    if selected_seed_movie_id:
                        final_recommendations_df, rec_method_used = get_hybrid_recommendations(selected_user, n=10, session_ratings=st.session_state.user_session_ratings, movie_id_seed=selected_seed_movie_id)
                    else:
                        st.error("Please rate movies to enable hybrid recommendations.")


                if not final_recommendations_df.empty:
                    st.info(f"Recommendations generated using: **{rec_method_used}**")
                    
                    # Display recommendations in a grid
                    cols_per_row_recs = 5
                    for i in range(0, len(final_recommendations_df), cols_per_row_recs):
                        cols = st.columns(cols_per_row_recs)
                        for j in range(cols_per_row_recs):
                            if i + j < len(final_recommendations_df):
                                movie = final_recommendations_df.iloc[i+j].to_dict()
                                with cols[j]:
                                    display_movie_card(movie, show_predicted_rating=True, reason_text=movie.get('reason', ''))
                else:
                    st.warning("Could not generate recommendations with the current settings. Try rating more movies or adjusting model type.")

        st.subheader(f"Your Recent Ratings (This Session)")
        if st.session_state.user_session_ratings:
            session_rated_movies_display = movies_df[movies_df['movieId'].isin(st.session_state.user_session_ratings.keys())].copy()
            session_rated_movies_display['user_rating'] = session_rated_movies_display['movieId'].map(st.session_state.user_session_ratings)
            # Sort by highest rating first
            session_rated_movies_display = session_rated_movies_display.sort_values('user_rating', ascending=False)
            
            for idx, row in session_rated_movies_display.iterrows():
                st.markdown(f"**{row['title']}** (Rated {row['user_rating']:.1f} ⭐)")
        else:
            st.info("You haven't rated any movies this session yet. Go to '🌟 Rate Movies' tab!")


st.markdown("""
<div class="app-footer">
    <p>Built with ❤️ and Streamlit. Data from simulated or provided CSV sources.</p>
</div>
""", unsafe_allow_html=True)