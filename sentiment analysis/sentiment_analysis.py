import pandas as pd
import re
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data1 = pd.read_csv('Twitter_Data.csv')
data2 = pd.read_csv('Reddit_Data.csv')

data2=data2.rename(columns={'clean_comment':'clean_text'})

data=pd.concat([data1, data2], axis=0, ignore_index=True)

data['category'].replace({1:'pos',0:'neu',-1:'neg'},inplace=True)
def preprocess_text(text):
    try:
        text = text.lower()  
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    except AttributeError:
        pass
    return text

data['cleaned_text'] = data['clean_text'].apply(preprocess_text)

data.dropna(inplace=True)
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def stop_w(text):
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)
data['clean_stop']=data['cleaned_text'].apply(stop_w)
def stem_w(text):
    stemmer = SnowballStemmer('english')
    words = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)
data['stemmed_text'] = data['clean_stop'].apply(stem_w)
X_train, X_test, y_train, y_test = train_test_split(data['stemmed_text'], data['category'], test_size=0.2, random_state=42)
vectorizer = CountVectorizer()

X_train_bow = vectorizer.fit_transform(X_train)

X_test_bow = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_bow, y_train)

y_pred = model.predict(X_test_bow)
new_text = "you are bad"
new_text_cleaned = preprocess_text(new_text)
new_text_cleaned1 = stop_w(new_text_cleaned)
new_text_cleaned2 = stem_w(new_text_cleaned1)
new_text_features = vectorizer.transform([new_text_cleaned2])

predicted_sentiment = model.predict(new_text_features)
print(f"Predicted Sentiment: {predicted_sentiment}")