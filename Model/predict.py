import pickle 
import nltk, string
from nltk.corpus import stopwords

MODEL_VERSION = "1.0.0"

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

ps = nltk.stem.porter.PorterStemmer()
stop_words = set(stopwords.words('english'))

cv = pickle.load(open('Model/vectorizer.pkl', 'rb'))
model = pickle.load(open('Model/model.pkl', 'rb'))

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

def predict_output(data):
    # 1. preprocess
    transformed_sms = text_transformed(data)
    # 2. vectorize
    cv_sms = cv.transform([transformed_sms])
    # 3. predict
    result = model.predict(cv_sms)[0]
    prob = model.predict_proba(cv_sms)

    return {
        "result":result,
        "probabilities":{
            "Spam":prob[0][0],
            "Not_Spam":prob[0][1]
        }
    }
