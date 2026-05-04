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

page_bg_img = """

<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fimg.freepik.com%2Fphotos-gratuite%2Fabstrait-numerique-grille-fond-noir_53876-97647.jpg%3Fsemt%3Dais_hybrid%26w%3D740&f=1&nofb=1&ipt=3a551da015306bbddd270ba2ae800e19c0cd7a495531591dc2fac4df03aeca4c");
    background-size: cover;
}

[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)

with st.sidebar:
    st.header("📱 SMS Spam Detector")

    with st.expander("About this app"):
        st.write("""
                This SMS Spam Detection app uses Machine Learning 
                to classify messages as Spam or Not Spam.
                 
                👉 Model: Logistic Regression  
                 
                👉 Vectorizer: CountVectorizer """)

    with st.expander("Limitations"):
        st.write("* Predictions may not be 100% accurate  \n\n* Model is still under improvement  \n\n* Avoid relying on this for critical decisions ")

    with st.expander("Future work"):
        st.write("- Better accuracy with TF-IDF  \n\n- Deep Learning models \n\n- Larger dataset training ")

def text_transformed(text):
    text=text.lower() # convert text to lowercase
    text= nltk.word_tokenize(text) # tokenize the text
    y=[]
    for i in text:
        if i.isalnum(): # remove special characters
            y.append(i)
    y2=[]
    for i in y:
        if i not in stop_words and i not in string.punctuation: # remove stop words and punctuation
            y2.append(ps.stem(i)) # stemming
    
    return " ".join(y2)

cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Spam Detection")
input_sms = st.text_area("Enter the SMS message:")

if st.button("Predict", type="primary"):
    # 1. preprocess 
    transformed_sms = text_transformed(input_sms)
    if transformed_sms.strip() == "":
        st.error("Please enter valid text")
        st.stop()
    # 2. vectorize
    cv_sms = cv.transform([transformed_sms])
    # 3. predict
    result = model.predict(cv_sms)[0]
    prob = model.predict_proba(cv_sms)

    col1,col2=st.columns([2,1])
    if result == 0:
        col1.error("Spam detected")
        col2.warning(f"Spam probability: {prob[0][0]:.2f} \n\n Not Spam probability: {prob[0][1]:.2f}")

    else:
        col1.success("Not Spam")
        col2.info(f"Spam probability: {prob[0][0]:.2f} \n\n Not Spam probability: {prob[0][1]:.2f}")


    