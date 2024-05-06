# Import the libraries
import pandas as pd
import numpy as np
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import pickle
import streamlit as st

# Define stopwords
stop_words = set(stopwords.words('english'))  


# Initialize SnowballStemmer
snowball_stemmer = SnowballStemmer(language='english')


# Unpickle model 
def unpickle_model(file_path):
    # Load the pickled model from the specified file
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
        return model
    

loaded_vectorizer = unpickle_model("models/vectorizer.pkl")
loaded_adaboost = unpickle_model("models/AdaBoost.pkl")




# Function to replace URLs with 'url' placeholder
def replace_urls(text):
    url_pattern = r'https?://\S+'
    return re.sub(url_pattern, 'url', text)


# Function to clean text
def clean_text(text, stop_words):
    # replace url in text
    text = replace_urls(text)
    # Remove punctuation using regular expression
    cleaned_text = re.sub(r"([^A-Za-z\s]+)", '', text)

    # Convert text to lowercase
    cleaned_text = cleaned_text.lower()

    # Remove stopwords
    cleaned_text = ' '.join(word for word in cleaned_text.split() if word not in stop_words)

    return cleaned_text


# Function to lemmatize text
def stem_text(text):

    # clean text
    text = clean_text(text, stop_words)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Stem each word in the text using SnowballStemmer
    snowball_stemmed_words = [snowball_stemmer.stem(word) for word in tokens]

    # Join the lemmatized tokens back into a string
    stemmed_text = ' '.join(snowball_stemmed_words)

    return stemmed_text


# Function to vectorize the text 
def vectorize_text(text_list):
    # clean text
    cleaned_texts = [stem_text(text) for text in text_list if text != ""]

    # vectorize text
    vectorized_text = loaded_vectorizer.transform(cleaned_texts)

    return vectorized_text


def label_sentiment(predict_sentiment):   
    # map the numeric prediction to text 
    if predict_sentiment == 1.0:
        return "Hate Speech"
    else:
        return "Not Hate Speech"


def main():
    st.set_page_config(
        page_title="Hate Speech Detector",
        page_icon=":rage:",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Define a dictionary to store the state
    state = {
        'user_input': ""
    }

    st.title("Hate Speech Detector Chatbot")

    # Some explanation about the app and the hate speech detection model
    st.markdown("This app detects whether text inputs contains hate speech or not.")

    # Text input for user to enter message
    user_input = st.text_area("Enter text here:", value=state['user_input'])
    state['user_input'] = user_input
    user_input = user_input.split("\n")


    # Create two columns 
    
    col1, col2 = st.columns(2)

    with col1:
        # Refresh button to clear the text area
        refresh_button = st.button("↻ Refresh")
        if refresh_button:
            state['user_input'] = ""

    predict_sentiment = []
    with col2:
        # Button to send the message
        if st.button("Detect Hate Speech"):
            if user_input: 
                with st.spinner("Detecting..."):
                # Clean and Vectorize user_input
                    word_vectorized = vectorize_text(user_input)
                    
                    # predict the sentiment of the user_input
                    predict_sentiment = loaded_adaboost.predict(word_vectorized)

            else:
                st.warning("Please enter some text.")
    
    # convert the predict from numberic to word
    for i in range(len(predict_sentiment)):
        st.success(f"Result: {label_sentiment(predict_sentiment[i])}")
    
    
    # Add a feedback mechanism section
    st.markdown("---")
    if st.button("Have a Feedback?"):
        st.subheader("Feedback")
        feedback = st.text_input("Have feedback or encountered an issue? Let us know!")
        if st.button("Submit Feedback"):
            # Handle feedback submission
            st.success("Thank you for your feedback!") 


           
if __name__ == "__main__":
    main()
