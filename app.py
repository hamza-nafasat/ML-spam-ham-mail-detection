import streamlit as st
import pickle
import nltk
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()

tfidf = pickle.load(open("vectroizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


# function which make text transformed
def transform_text(text):
    # make the all text lower case
    text = text.lower()
    # tokenization
    text = nltk.word_tokenize(text)
    # remove special characters only alpha & numerics remaining
    modifiedText = []
    for i in text:
        if i.isalnum():
            modifiedText.append(i)
    text = modifiedText[:]
    modifiedText.clear()
    # removing stop words and punctuations
    stop_words = stopwords.words("english")
    punctuations = string.punctuation
    for i in text:
        if i not in stop_words and i not in punctuations:
            modifiedText.append(i)
    text = modifiedText[:]
    modifiedText.clear()
    # seeming the words
    for i in text:
        modifiedText.append(ps.stem(i))
    return " ".join(modifiedText)


st.title("Email/SMS Spam Classifier")
input_sms = st.text_area(
    "Enter your email and check it is spam or not...?", "", height=300
)


if st.button("Predict"):
    # 1.preprocess
    transformed_text = transform_text(input_sms)
    #      2.vectorize
    vectorInput = tfidf.transform([transformed_text])
    # 3.predict
    result = model.predict(vectorInput)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


# run this code with streamilit run app.py
