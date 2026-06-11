import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#Load the IMDB dataset word index
word_index = imdb.get_word_index()

#Load pre-trained model
model = load_model('imdb-sentiment-analysis.h5')


#function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()

    encoded_review = []
    for word in words:
      encoded_review.append(word_index.get(word , 2) + 3)


    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen = 500
    )
    return padded_review


#prediction function
def predict_sentiment(review):
  preprocessed_input = preprocess_text(review)

  prediction = model.predict(preprocessed_input)

  if prediction[0][0] > 0.5:
    sentiment = 'Positive'
  else:
    sentiment = 'Negative'

  return sentiment , prediction[0][0]


#streamlit app
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

#user input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=preprocess_text(user_input)

    ## Make prediction
    prediction=model.predict(preprocessed_input)
    if prediction[0][0] > 0.5:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')
