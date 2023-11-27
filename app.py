from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
import pickle

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

class Text(BaseModel):
    input: str

def preprocess_text(text):
    punc = list(punctuation)
    stop = stopwords.words('english')
    bad_tokens = punc + stop
    lemma = WordNetLemmatizer()
    tokens = word_tokenize(text)
    word_tokens = [t for t in tokens if t.isalpha()]
    clean_tokens = [lemma.lemmatize(t.lower()) for t in word_tokens if t not in bad_tokens]
    return ' '.join(t for t in clean_tokens)

@app.get("/")
def index():
    return {"Hello": "there!!"}

@app.post("/predict/")
def predict(text: Text):
    data = text.dict()
    text = data['input']
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(vectorized_text)[0]

    if prediction == 1:
        return {"Language": "English"}
    if prediction == 2:
        return {"Language": "Malayalam"}
    if prediction == 3:
        return {"Language": "Hindi"}
    if prediction == 4:
        return {"Language": "Tamil"}
    if prediction == 5:
        return {"Language": "Portuguese"}
    if prediction == 6:
        return {"Language": "French"}
    if prediction == 7:
        return {"Language": "Dutch"}
    if prediction == 8:
        return {"Language": "Spanish"}
    if prediction == 9:
        return {"Language": "Greek"}
    if prediction == 10:
        return {"Language": "Russian"}
    if prediction == 11:
        return {"Language": "Danish"}
    if prediction == 12:
        return {"Language": "Italian"}
    if prediction == 13:
        return {"Language": "Turkish"}
    if prediction == 14:
        return {"Language": "Swedish"}
    if prediction == 15:
        return {"Language": "Arabic"}
    if prediction == 16:
        return {"Language": "German"}
    if prediction == 17:
        return {"Language": "Kannada"}
