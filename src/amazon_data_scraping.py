import pandas as pd
import re
import string
import nltk
from google_play_scraper import reviews, Sort

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


############################################################

reviews_gp, _ = reviews(
    'com.amazon.mShop.android.shopping',
    lang='en',
    country='us',
    sort=Sort.NEWEST,
    count=500
)

df_gp = pd.DataFrame(reviews_gp)
df_gp = df_gp[['content', 'score']]
df_gp.columns = ['commentaire', 'rating']
df_gp['source'] = 'Google Play'

df_gp.head()

print(df_gp)

############################################################

df_gp.to_csv("amazon_google_play_reviews.csv", index=False, encoding="utf-8")

###############################################################

df = pd.read_csv("amazon_google_play_reviews.csv")
df.head()

################################################3################

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(tokens)


#####################################################################

df["clean_text"] = df["commentaire"].apply(preprocess_text)

df = df[df["clean_text"].str.strip() != ""]
df = df.drop_duplicates(subset="clean_text")

df[["commentaire", "clean_text"]].head()

###########################################################################

def rating_to_sentiment(r):
    if r <= 2:
        return "negative"
    elif r == 3:
        return "neutral"
    else:
        return "positive"

df["sentiment"] = df["rating"].apply(rating_to_sentiment)

df[["rating", "sentiment"]].head()

#####################################################################


tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = tfidf.fit_transform(df["clean_text"])
y = df["sentiment"]

X.shape, y.shape

########################################################################


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


##########################################################################

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

###################################################################

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


#################################################################


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["negative", "neutral", "positive"]
)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["negative", "neutral", "positive"],
    yticklabels=["negative", "neutral", "positive"]
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

###################################################################

def predict_sentiment(text):
    clean = preprocess_text(text)
    vec = tfidf.transform([clean])
    return model.predict(vec)[0]

predict_sentiment("The app is very slow and delivery is terrible")