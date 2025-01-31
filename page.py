import streamlit as st
import numpy as np
import gdown
import pickle
from pathlib import Path
import os
from pathlib import Path
from safetensors.torch import load_file
import joblib

# Custom CSS for star rating
st.markdown("""
<style>
.rating {
    display: inline-block;
    font-size: 30px;
    cursor: pointer;
}
.rating input {
    display: none;
}
.rating label {
    color: #ddd;
    float: right;
    padding: 0 2px;
    cursor: pointer;
}
.rating label:before {
    content: '★';
}
.rating input:checked ~ label {
    color: #ffdd00;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'sentiment_score' not in st.session_state:
    st.session_state.sentiment_score = 0
if 'ratings' not in st.session_state:
    st.session_state.ratings = {}
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Function to load the model
MODEL_PATH = "/workspaces/Tip_recomendation/svm_model.pkl"

def load_model():
    try:
        with st.spinner('Loading model...'):
            model = joblib.load(MODEL_PATH)  # Load using joblib
            st.session_state.model = model
            st.session_state.model_loaded = True
            return True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return False

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if not st.session_state.model_loaded:
    load_model()

def analyze_sentiment(review):
    if st.session_state.model_loaded:
        try:
            prediction = st.session_state.model.predict([review])
            return prediction[0]
        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
            return 0
    return 0

def calculate_tip_percentage(avg_rating, sentiment_score):
    return (avg_rating * 2) + (sentiment_score * 5)

# Custom star rating component
def star_rating(key, label):
    col1, col2 = st.columns([3, 2])
    with col1:
        st.write(label)
    with col2:
        rating = st.select_slider(
            f"Rating for {key}",
            options=[1, 2, 3, 4, 5],
            value=3,
            label_visibility="collapsed",
            format_func=lambda x: "★" * x + "☆" * (5 - x)
        )
    return rating

# Main app
st.title("✨ Restaurant Tip Recommendation System ✨")

# Progress bar
if st.session_state.step > 1:
    progress = (st.session_state.step - 1) / 3
    st.progress(progress)

# Step 1: Review Input
if st.session_state.step == 1:
    st.header("📝 Step 1: Share Your Experience")
    
    # Add some context
    st.markdown("""
    Please share your dining experience with us. Your feedback helps us provide 
    better service and an accurate tip recommendation.
    """)
    
    review = st.text_area(
        "How was your experience?",
        height=150,
        placeholder="Tell us about your dining experience..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Submit Review", use_container_width=True):
            if review:
                with st.spinner('Analyzing your review...'):
                    st.session_state.sentiment_score = analyze_sentiment(review)
                    sentiment = "Positive 😊" if st.session_state.sentiment_score == 1 else "Negative ☹️"
                    st.success(f"Review sentiment: {sentiment}")
                    st.session_state.step = 2

            else:
                st.warning("Please write a review before submitting.")

# Step 2: Ratings
elif st.session_state.step == 2:
    st.header("⭐ Step 2: Rate Your Experience")
    
    rating_questions = {
        'food_quality': 'Food Quality – Taste, freshness, and presentation',
        'service_quality': 'Service Quality – Staff attentiveness and friendliness',
        'cleanliness': 'Cleanliness – Restaurant environment',
        'ambiance': 'Ambiance – Overall atmosphere and comfort',
        'value_for_money': 'Value for Money – Experience worth the price'
    }
    
    st.markdown("### Please rate each aspect of your experience:")
    
    for key, question in rating_questions.items():
        st.session_state.ratings[key] = star_rating(key, question)
        st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Submit Ratings", use_container_width=True):
            st.session_state.step = 3


# Step 3: Bill Amount
elif st.session_state.step == 3:
    st.header("💰 Step 3: Enter Bill Amount")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        bill_amount = st.number_input(
            "Bill Amount ($)",
            min_value=0.0,
            step=0.01,
            format="%.2f"
        )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Calculate Tip", use_container_width=True):
            if bill_amount > 0:
                # Calculate results
                avg_rating = np.mean(list(st.session_state.ratings.values()))
                tip_percentage = calculate_tip_percentage(avg_rating, st.session_state.sentiment_score)
                tip_amount = (bill_amount * tip_percentage) / 100
                final_amount = bill_amount + tip_amount
                
                # Display results in a nice format
                st.markdown("---")
                st.markdown("### 📊 Tip Recommendation Summary")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentiment", "Positive 😊" if st.session_state.sentiment_score == 1 else "Negative ☹️")
                    st.metric("Average Rating", f"{avg_rating:.1f}/5 ⭐")
                    st.metric("Tip Percentage", f"{tip_percentage:.1f}%")
                
                with col2:
                    st.metric("Bill Amount", f"${bill_amount:.2f}")
                    st.metric("Tip Amount", f"${tip_amount:.2f}")
                    st.metric("Final Amount", f"${final_amount:.2f}")
                
                # Add some visual separation
                st.markdown("---")
                
                # Start over button
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("Start Over", use_container_width=True):
                        st.session_state.step = 1
                        st.session_state.ratings = {}
                        st.session_state.sentiment_score = 0
                        st.experimental_rerun()
            else:
                st.warning("Please enter a valid bill amount.")

# Add a footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: grey;'>
        Made with ❤️ | Restaurant Tip Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)