### 📩 SMS Spam Detection App

live link:
This is a simple SMS Spam Detection web app built using Streamlit, Machine Learning, and Natural Language Processing (NLP). The app takes a text message as input and predicts whether it is Spam or Not Spam.

⸻

🚀 Features

* Detects spam messages in real time
* Uses NLP techniques for text preprocessing
* Lightweight and easy-to-use interface
* Built with a trained machine learning model

⸻

🧠 How It Works

The app follows these steps:

1. Text Preprocessing
    * Converts text to lowercase
    * Tokenizes the message
    * Removes special characters
    * Removes stopwords
    * Applies stemming
2. Vectorization
    * Transforms text using a pre-trained vectorizer (CountVectorizer)
3. Prediction
    * Uses a trained ML model (Logistic Regression or similar)
    * Outputs:
        * Spam
        * Not Spam

⸻

🛠️ Tech Stack

* Python
* Streamlit
* scikit-learn
* NLTK
* Pickle

⸻
📌 Usage

1. Enter an SMS message in the text box
2. Click on Predict
3. The app will display:
    * ⚠️ Spam
    * ✅ Not Spam

⸻

📊 Model Details

* Text transformed using NLP preprocessing
* Vectorized using CountVectorizer
* Model trained on SMS spam dataset

⸻

⚠️ Note

* Make sure model.pkl and vectorizer.pkl are present in the project directory
* NLTK stopwords are downloaded automatically
