import streamlit as st
import requests

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

API_url="http://127.0.0.1:8000/predict"

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


st.title("SMS Spam Detection")
input_sms = st.text_area("Enter the SMS message:")
text = {"data": input_sms}

if st.button("Predict", type="primary"):
     
    if input_sms.strip() == "" :
        st.error("Please enter valid text")
        st.stop()

    prediction=None
    try:
        response = requests.post(API_url, json=text, timeout=10)
        if response.status_code == 200:
            prediction = response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    col1,col2=st.columns([2,1])
    if prediction['result'] == 0:
        col1.error("Spam detected")
    else:
        col1.success("Not Spam")
        
    col2.warning(f"Spam probability: {prediction['probabilities']['Spam']:.2f} \n\n Not Spam probability: {prediction['probabilities']['Not_Spam']:.2f}")


# testing data
test_text1 = """Accident compensation 
You have still not claimed the compensation you are due for the accident you had. 
To start the process please reply YES. To opt out text STOP"""

test_text2 = """A [redacted] Loan for £950 is approved for you if you receive this SMS. 
1 min verification & cash in 1hr at www.[redacted].co.uk to opt out reply stop"""

test_text3 = """I am free today, lets go out for a movie. What do you say?"""

test_text4 = """You could be entitled up to £3,160 in compensation from mis-sold PPI 
on a credit card or Loan. Please reply PPI for info or STOP to opt out."""

test_text5 = """congratulations you won 1000 call on thist number to get your prize"""

col1,col2=st.columns([1,1])
with col1.expander("Test the model with sample messages"):
    
    st.code(test_text1, language="text")
    st.code(test_text2, language="text")
    st.code(test_text3, language="text")
    st.code(test_text4, language="text")
