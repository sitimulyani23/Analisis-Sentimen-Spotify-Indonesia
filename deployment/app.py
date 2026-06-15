import gradio as gr
import pickle

# Load model dan TF-IDF
model = pickle.load(open("model_logreg.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Fungsi prediksi
def predict_sentiment(review):

    review_tfidf = tfidf.transform([review])

    prediction = model.predict(review_tfidf)[0]

    return prediction


# Interface Gradio
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Masukkan ulasan Spotify...",
        label="Ulasan"
    ),
    outputs=gr.Textbox(label="Hasil Prediksi"),
    title="Analisis Sentimen Spotify Indonesia",
    description="Prediksi sentimen ulasan Spotify menggunakan Logistic Regression."
)

demo.launch()
