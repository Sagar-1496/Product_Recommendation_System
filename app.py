import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
nltk.download('punkt')

df_train = pd.read_csv("/Users/sagarpatel/Flipkart_Product_Recommendation/train.csv")

# Forward fill null values in 'maincateg' column
df_train['maincateg'].fillna(method='ffill', inplace=True)

remove = ['norating1', 'noreviews1', 'star_5f', 'star_4f', 'star_3f']
# Drop specified columns
df_train.drop(columns = remove, inplace=True)

def tokenize_stem(text):
    ps = SnowballStemmer('english')
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return stemmed_tokens

df_train['stemmed_tokens'] = df_train['title'].apply(tokenize_stem)

tfidfv = TfidfVectorizer(tokenizer=tokenize_stem)
def cosine_sim(txt1,txt2):
    matrix = tfidfv.fit_transform([txt1,txt2])
    return cosine_similarity(matrix)

def search_product(query):
    stemmed_query = tokenize_stem(query)
     # Compute cosine similarity between query and each product title
    df_train['similarity'] = df_train['stemmed_tokens'].apply(lambda x: cosine_sim(' '.join(stemmed_query), ' '.join(x))[0][0])
     # Weight the similarity by ratings
    df_train['weighted_similarity'] = df_train['similarity'] * df_train['Rating']
    res = df_train.sort_values(by=['similarity'], ascending=False).head(10)[['title', 'Rating']]
    return res

#Web app
img = Image.open('/Users/sagarpatel/Flipkart_Product_Recommendation/flipkart-logo-3F33927DAA-seeklogo.com.png')
st.image(img,width=900)
st.title("Product Recommendation System on Flipkart Data")
query = st.text_input("Enter Product Name")
submit = st.button('Search')
if submit:
    result = search_product(query)
    st.write(result)
