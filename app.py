"""
app.py
Streamlit web application for Amazon Product Review Sentiment Analysis.

Run with:  streamlit run app.py
"""

import os
import pandas as pd
import streamlit as st

from predict import predict_single_review, predict_batch_reviews
from visualization import generate_all_visualizations
from utils import ensure_dir

OUTPUT_CSV_PATH = "outputs/prediction.csv"
GRAPH_DIR = "outputs/graphs"

st.set_page_config(page_title="Amazon Review Sentiment Analysis", page_icon="🛒", layout="wide")

st.title("🛒 Amazon Product Review Sentiment Analysis")
st.write("Analyze the sentiment of Amazon product reviews using a trained Machine Learning model.")

mode = st.sidebar.radio(
    "Choose Input Type",
    ("Predict a Single Review", "Upload a CSV File"),
)

# ---------------------------------------------------------------
# MODE 1: Single review prediction
# ---------------------------------------------------------------
if mode == "Predict a Single Review":
    st.subheader("🔹 Single Review Sentiment Prediction")
    review_text = st.text_area(
        "Enter an Amazon product review:",
        placeholder="e.g. The product quality is amazing and delivery was fast.",
        height=120,
    )

    if st.button("Predict Sentiment"):
        if not review_text.strip():
            st.warning("Please enter a review before predicting.")
        else:
            try:
                sentiment = predict_single_review(review_text)
                color = {"Positive": "green", "Negative": "red", "Neutral": "orange"}[sentiment]
                st.markdown(f"### Predicted Sentiment: :{color}[{sentiment}]")
            except FileNotFoundError:
                st.error("Model not found. Please run `python train_model.py` first to train the model.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# ---------------------------------------------------------------
# MODE 2: CSV batch prediction
# ---------------------------------------------------------------
else:
    st.subheader("🔹 Batch Review Sentiment Prediction (CSV Upload)")
    uploaded_file = st.file_uploader("Upload a CSV file containing a 'review' column", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV file: {e}")
            df = None

        if df is not None:
            if "review" not in df.columns:
                st.error("CSV file must contain a column named 'review'.")
            else:
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())

                if st.button("Run Batch Prediction"):
                    try:
                        with st.spinner("Predicting sentiments for all reviews..."):
                            result_df = predict_batch_reviews(df, text_column="review")

                            ensure_dir(os.path.dirname(OUTPUT_CSV_PATH))
                            result_df.to_csv(OUTPUT_CSV_PATH, index=False)

                        st.success("Prediction complete!")

                        # ----- Summary statistics -----
                        total = len(result_df)
                        counts = result_df["predicted_sentiment"].value_counts()
                        pos = int(counts.get("Positive", 0))
                        neg = int(counts.get("Negative", 0))
                        neu = int(counts.get("Neutral", 0))

                        st.subheader("📊 Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Reviews", total)
                        col2.metric("Positive", f"{pos} ({pos/total*100:.1f}%)")
                        col3.metric("Negative", f"{neg} ({neg/total*100:.1f}%)")
                        col4.metric("Neutral", f"{neu} ({neu/total*100:.1f}%)")

                        # ----- Visualizations -----
                        st.subheader("📈 Visualizations")
                        graph_paths = generate_all_visualizations(result_df, GRAPH_DIR)

                        g1, g2 = st.columns(2)
                        with g1:
                            if "bar_chart" in graph_paths:
                                st.image(graph_paths["bar_chart"], caption="Sentiment Bar Chart")
                            if "count_plot" in graph_paths:
                                st.image(graph_paths["count_plot"], caption="Sentiment Count Plot")
                        with g2:
                            if "pie_chart" in graph_paths:
                                st.image(graph_paths["pie_chart"], caption="Sentiment Pie Chart")
                            if "length_distribution" in graph_paths:
                                st.image(graph_paths["length_distribution"], caption="Review Length Distribution")

                        st.subheader("🔤 Most Common Words by Sentiment")
                        w1, w2, w3 = st.columns(3)
                        if "top_words_positive" in graph_paths:
                            w1.image(graph_paths["top_words_positive"], caption="Positive")
                        if "top_words_negative" in graph_paths:
                            w2.image(graph_paths["top_words_negative"], caption="Negative")
                        if "top_words_neutral" in graph_paths:
                            w3.image(graph_paths["top_words_neutral"], caption="Neutral")

                        # ----- Download button -----
                        st.subheader("⬇️ Download Results")
                        csv_data = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download Predictions CSV",
                            data=csv_data,
                            file_name="prediction.csv",
                            mime="text/csv",
                        )

                        st.dataframe(result_df[["review", "predicted_sentiment"]].head(20))

                    except FileNotFoundError:
                        st.error("Model not found. Please run `python train_model.py` first to train the model.")
                    except Exception as e:
                        st.error(f"An error occurred during batch prediction: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Default model: Logistic Regression + TF-IDF")