import streamlit as st
import numpy as np
import gdown
import os
import requests
import torch
import transformers
from safetensors.torch import load_file

# Custom CSS for styling
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

# Google Drive file configuration
GOOGLE_DRIVE_FILE_ID = "1X0nR3EXjMVQn1XpeY2WhIf6oETw_xcpe"
MODEL_PATH = "model.safetensors"

# Advanced download function with detailed error handling
@st.cache_resource()
def download_model():
    try:
        # Check if model already exists
        if not os.path.exists(MODEL_PATH):
            with st.spinner('Downloading model...'):
                # Multiple download attempt strategies
                download_strategies = [
                    lambda: gdown.download(f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}', MODEL_PATH, quiet=False),
                    lambda: os.system(f'wget --no-check-certificate "https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}" -O {MODEL_PATH}')
                ]
                
                for strategy in download_strategies:
                    try:
                        result = strategy()
                        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
                            return True
                    except Exception as e:
                        st.warning(f"Download strategy failed: {str(e)}")
                
                raise Exception("All download strategies failed")
        
        return True
    except Exception as e:
        st.error(f"Comprehensive Download Error: {str(e)}")
        st.error("Please manually download the model from the Google Drive link.")
        return False

# Cached model loading with extensive error handling
@st.cache_resource()
def load_model():
    try:
        with st.spinner('Loading model...'):
            # Validate file exists and has content
            if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
                raise FileNotFoundError(f"Model file {MODEL_PATH} is missing or empty")
            
            # Load the SafeTensors model
            state_dict = load_file(MODEL_PATH)
            
            # Dynamically identify model architecture
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                'distilbert-base-uncased', 
                num_labels=2  # Binary classification
            )
            
            # Load state dict
            model.load_state_dict(state_dict)
            
            # Load tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
            
            return model, tokenizer
    
    except Exception as e:
        st.error(f"Model Loading Error: {str(e)}")
        return None, None

# Sentiment analysis function with robust error handling
def analyze_sentiment(review, model, tokenizer):
    if model and tokenizer:
        try:
            # Tokenize the input
            inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Perform prediction
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=1).item()
            
            # Return 1 for positive, 0 for negative
            return 1 if predicted_class == 1 else 0
        
        except Exception as e:
            st.error(f"Sentiment Analysis Error: {str(e)}")
            return 0
    return 0

# Tip calculation function
def calculate_tip_percentage(avg_rating, sentiment_score):
    base_tip = (avg_rating * 2) + (sentiment_score * 5)
    return max(0, min(base_tip, 25))  # Cap tip between 0% and 25%

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

# Initialize session state
def initialize_session_state():
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'sentiment_score' not in st.session_state:
        st.session_state.sentiment_score = 0
    if 'ratings' not in st.session_state:
        st.session_state.ratings = {}
    
    # Download and load model
    if download_model():
        st.session_state.model, st.session_state.tokenizer = load_model()
        st.session_state.model_loaded = st.session_state.model is not None

# Main Streamlit app
def main():
    # Initialize session state
    initialize_session_state()

    # App title
    st.title("✨ Restaurant Tip Recommendation System ✨")

    # System Status Check
    if not st.session_state.get('model_loaded', False):
        st.error("⚠️ Model not loaded. Please check download and setup.")
        st.markdown("""
        ### Troubleshooting Steps:
        1. Ensure stable internet connection
        2. Verify Google Drive link
        3. Check file permissions
        4. Manually download the model file
        """)
        return

    # Progress bar
    if st.session_state.step > 1:
        progress = (st.session_state.step - 1) / 3
        st.progress(progress)

    # Step 1: Review Input
    if st.session_state.step == 1:
        st.header("📝 Step 1: Share Your Experience")
        
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
                        st.session_state.sentiment_score = analyze_sentiment(
                            review, 
                            st.session_state.model,
                            st.session_state.tokenizer
                        )
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

# Run the app
if __name__ == "__main__":
    main()