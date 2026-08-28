import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('spam_model.pkl','rb'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

st.set_page_config(page_title="Spam Detector")
st.title("Spam SMS Detector")
st.write("NLP Project using TF-IDF and Naive Bayes | Accuracy: 97.58%")
st.write("Enter a message to classify it as Spam or Not Spam.")

msg = st.text_area("Message:", "Congratulations! You have won $1000. Click here to claim.")

if st.button("Predict"):
    cleaned = clean_text(msg)
    vect = tfidf.transform([cleaned])
    pred = model.predict(vect)[0]
    if pred == 1:
        st.error("Prediction: SPAM")
        st.write("This message is classified as spam.")
    else:
        st.success("Prediction: NOT SPAM")