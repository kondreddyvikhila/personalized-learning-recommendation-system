%%writefile data_preprocessing.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("final_courses_dataset.csv")

df.fillna("", inplace=True)

df["combined_features"] = df["title"] + " " + df["skills"] + " " + df["level"]

tfidf = TfidfVectorizer(stop_words="english")
matrix = tfidf.fit_transform(df["combined_features"])

cosine_sim = cosine_similarity(matrix, matrix)

indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

def recommend(title, top_n=5):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    course_idx = [i[0] for i in sim_scores]

    return df.iloc[course_idx][["title","platform","level","rating"]]
