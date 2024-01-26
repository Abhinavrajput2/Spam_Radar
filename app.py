import streamlit as st
import sklearn
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('PorterStemmer')

ps=PorterStemmer()
def load_model(path='Model.pkl'):
    with open(path,'rb') as f:
        return pickle.load(f)
    
def lode_tfidf(path='vectorizer.pkl'):
    with open(path,'rb') as j:
      return pickle.load(j)

# tfidf = pickle.load(open('vectorizer.pkl','rb'))
    
# model = pickle.load(open('model.pkl','rb'))
st.title("email")
input_msg = st.text_area("Enter the msg")

def text_prepro(text):

  text = text.lower()
  text = nltk.word_tokenize(text)

  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)

  text=y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words("english") and i not in string.punctuation:
      y.append(i)

  text=y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

if st.button("predict"):
    #1. preprocessing
    transform_text = text_prepro(input_msg)
    #2. vectorize
    vectorizer = lode_tfidf() 
    vector_input = vectorizer.transform([transform_text])
    try:
        # 3. Predict
        model =load_model()
        
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    except Exception as e:
        st.error(f"Error: {str(e)}. Make sure the model is fitted with training data.")