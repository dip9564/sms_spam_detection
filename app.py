import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
import string
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
ps = nltk.stem.porter.PorterStemmer()

def text_transformed(text):
    text=text.lower() # convert text to lowercase
    text= nltk.word_tokenize(text) # tokenize the text
    y=[]
    for i in text:
        if i.isalnum(): # remove special characters
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation: # remove stop words and punctuation
            y.append(ps.stem(i)) # stemming
    
    return " ".join(y)

cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Spam Detection")
input_sms = st.text_area("Enter the SMS message:")

# 1. preprocess 
transformed_sms = text_transformed(input_sms)
# 2. vectorize
cv_sms = cv.transform([transformed_sms])
# 3. predict
result = model.predict(cv_sms)[0]

if st.button("Predict"):
    if result == 0:
        st.warning("Spam")
    else:
        st.success("Not Spam")