import streamlit as st
import nltk
from nltk.corpus import opinion_lexicon
import string

# Download required NLTK data
nltk.download('opinion_lexicon', quiet=True)

# Load positive and negative words
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

# Preprocessing
def preprocess(text):
    text = text.lower()
    # text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

# Sentiment analysis with percentage output
def analyze_sentiment(text):
    words = preprocess(text)
    
    positive_found = [word for word in words if word in positive_words]
    negative_found = [word for word in words if word in negative_words]

    pos_count = len(positive_found)
    neg_count = len(negative_found)
    total = pos_count + neg_count

    if total > 0:
        pos_percent = (pos_count / total) * 100
        neg_percent = (neg_count / total) * 100
    else:
        pos_percent = neg_percent = 0

    if pos_count > neg_count:
        sentiment = "Positive"
    elif neg_count > pos_count:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, round(pos_percent, 2), round(neg_percent, 2), positive_found, negative_found


# Streamlit UI
st.title("Simple Sentiment Analyzer")
st.subheader("Enter a sentence to analyze the balance of positive and negative sentiment.")

user_input = st.text_input("Enter your statement:")

# Input box
user_input = st.text_input("Enter your statement:")

# Button to trigger analysis
if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        sentiment, pos_pct, neg_pct, pos_words, neg_words = analyze_sentiment(user_input)

        st.markdown(f"### Sentiment: **{sentiment}**")
        st.progress(int(pos_pct), text=f"Positive: {pos_pct}%")
        st.progress(int(neg_pct), text=f"Negative: {neg_pct}%")

        st.write(f"**Positive Percent:** {pos_pct}%")
        st.write(f"**Negative Percent:** {neg_pct}%")

        st.write(f"**Positive Words:** {pos_words}")
        st.write(f"**Negative Words:** {neg_words}")

        if pos_pct > neg_pct:
            st.subheader("The statement is more positively inclined.")
        elif pos_pct < neg_pct:
            st.subheader("The statement is more negatively inclined.")
        else:
            st.subheader("The statement is neutral.")
    else:
        st.warning("Please enter a statement before clicking the button.")

