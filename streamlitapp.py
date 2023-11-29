from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.write("# Language Detection System")

inputt = st.text_area("Enter text here")

def preprocess_text(text):
    punc = list(punctuation)
    stop = stopwords.words('english')
    bad_tokens = punc + stop
    lemma = WordNetLemmatizer()
    tokens = word_tokenize(text)
    word_tokens = [t for t in tokens if t.isalpha()]
    clean_tokens = [lemma.lemmatize(t.lower()) for t in word_tokens if t not in bad_tokens]
    return ' '.join(t for t in clean_tokens)

if st.button("Detect Language"):
    processed_text = preprocess_text(inputt)
    vectorized = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        st.header("English")
    if prediction == 2:
        st.header("Malayalam")
    if prediction == 3:
        st.header("Hindi")
    if prediction == 4:
        st.header("Tamil")
    if prediction == 5:
        st.header("Portuguese")
    if prediction == 6:
        st.header("French")
    if prediction == 7:
        st.header("Dutch")
    if prediction == 8:
        st.header("Spanish")
    if prediction == 9:
        st.header("Greek")
    if prediction == 10:
        st.header("Russian")
    if prediction == 11:
        st.header("Danish")
    if prediction == 12:
        st.header("Italian")
    if prediction == 13:
        st.header("Turkish")
    if prediction == 14:
        st.header("Swedish")
    if prediction == 15:
        st.header("Arabic")
    if prediction == 16:
        st.header("German")
    if prediction == 17:
        st.header("Kannada")