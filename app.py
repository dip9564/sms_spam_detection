import streamlit as st
import pickle
from nltk.corpus import stopwords
import string,nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = nltk.stem.porter.PorterStemmer()
stop_words = set(stopwords.words('english'))

def text_transformed(text):
    text=text.lower() # convert text to lowercase
    text= nltk.word_tokenize(text) # tokenize the text
    y=[]
    for i in text:
        if i.isalnum(): # remove special characters
            y.append(i)
    y2=[]
    for i in text:
        if i not in stop_words and i not in string.punctuation: # remove stop words and punctuation
            y2.append(ps.stem(i)) # stemming
    
    return " ".join(y2)

cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Spam Detection")
input_sms = st.text_area("Enter the SMS message:")


if st.button("Predict"):
    # 1. preprocess 
    transformed_sms = text_transformed(input_sms)
    # 2. vectorize
    cv_sms = cv.fit_transform([transformed_sms])
    # 3. predict
    result = model.predict(cv_sms)[0]

    if result == 0:
        st.warning("Spam")
    else:
        st.success("Not Spam")

    prob = model.predict_proba(cv_sms)
    st.write("Confidence:", prob)