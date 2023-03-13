import streamlit as st  
import tensorflow as tf
import re
import numpy as np 
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from PIL import Image
img = Image.open(r"E:\ABSA\images.jfif")
st.image(img, width=300)

st.header("ASPECT BASED SENTIMENT ANALYSIS")
model=tf.keras.models.load_model(r'E:\ABSA\ABSA.hdf5')
model1=tf.keras.models.load_model(r'E:\ABSA\aspect.hdf5')

message = st.text_area("Enter Review")
if st.button("Analyze"):
    with st.spinner("Analyzing the text …"):
        tokenizer=pickle.load(open('E:/ABSA/tokenizer.pickle','rb'))
        firstName, lastName = message.split(',')
        st.write("First Sentence: ",firstName)
        st.write("Second Sentence: ",lastName)
        sequence = tokenizer.texts_to_sequences([firstName])
        test = pad_sequences(sequence, maxlen=200)
        Aspect = ['Service' ,'Other', 'Food', 'Price' ,'Ambience']
        result1 = Aspect[np.round(model1.predict(test), decimals=0).argmax(axis=1)[0]]
        st.write("First Sentence Aspect: ",result1)

        sentiment = ['Neutral','Negative','Positive']
        result = sentiment[np.round(model.predict(test), decimals=0).argmax(axis=1)[0]]
        st.write("First Sentence Sentiment: ", result)
     
        sequence1 = tokenizer.texts_to_sequences([lastName])
        test1 = pad_sequences(sequence1, maxlen=200)
        Aspect1 = ['Service' ,'Other', 'Food', 'Price' ,'Ambience']
        result2 = Aspect1[np.round(model1.predict(test1), decimals=0).argmax(axis=1)[0]]
        st.write("Second Sentence Aspect: ",result2)
        sentiment1 = ['Neutral','Negative','Positive']
        result1 = sentiment1[np.round(model.predict(test1), decimals=0).argmax(axis=1)[0]]
        st.write("Second Sentence Sentiment: ", result1)
        
 
 


