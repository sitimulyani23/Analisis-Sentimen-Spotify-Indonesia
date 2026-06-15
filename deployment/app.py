import streamlit as st
import pickle

# Load model dan TF-IDF
model = pickle.load(open("model_logreg.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("Analisis Sentimen Spotify Indonesia")

st.write("Masukkan ulasan Spotify untuk diprediksi sentimennya")

review = st.text_area("Masukkan Ulasan")

if st.button("Prediksi"):

    if review.strip() != "":

        review_tfidf = tfidf.transform([review])

        prediction = model.predict(review_tfidf)[0]

        if prediction == "positif":
            st.success("Sentimen Positif 😊")

        elif prediction == "negatif":
            st.error("Sentimen Negatif 😠")

        else:
            st.warning("Sentimen Netral 😐")
