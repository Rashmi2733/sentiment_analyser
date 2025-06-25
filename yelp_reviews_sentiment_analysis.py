'''
It's been a few years since heard of 'sentiment analysis' but having built a small version myself in the past few days, I see and understand it more now. 


Used: Ml model SVM

TfidfVectorizer

serpAPI to get yelp reviews 


This doesnt really solve much currently but i see the potential -- looking up a keyword on a specific social media platform and seeing if it is positively or negatively being talked about 

'''

#getting all the necessary libraries and functions
from serpapi import GoogleSearch
import pickle
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os


##Getting the serp api key from .env file
load_dotenv() 
# SERP_API = os.getenv("SERP_API")

SERP_API = st.secrets["SERP_API"]

##Setting the Streamlit UI to be wide since the default is narrower
st.set_page_config(layout="wide")


current_dir = os.path.dirname(os.path.abspath(__file__))

##Getting the list of businesses (and their information) from business name and location from yelp (using Serp API)
def search_businesses(business_name: str, location: str):
    params = {
        "api_key": SERP_API,
        "engine": "yelp",
        "find_desc": business_name,
        "find_loc": location,
        "start": 0
    }
    search = GoogleSearch(params)
    data = search.get_dict()
    return data.get("organic_results", [])

##getting top 10 reviews from place ids of the business
def get_reviews(place_id: str, num_reviews: int = 10):
    params = {
        "api_key": SERP_API,
        "engine": "yelp_reviews",
        "place_id": place_id,
        "start": 0,
        "num": num_reviews
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("reviews", [])

#Loading the required models for svm and the vectorizer as well
def load_models():
    svm_path = os.path.join(current_dir, 'svm_sentiment_analyser.pkl')
    vec_path = os.path.join(current_dir, 'vectorizer.pkl')
    svm_model = pickle.load(open(svm_path, 'rb'))
    vect = pickle.load(open(vec_path, 'rb'))

    return svm_model, vect

svm_model, vectorizer = load_models()


##Streamlit code

#Adding a self made yelp logo 
st.sidebar.image('yelp.jpeg', width = 60)
##Showing the heading in the sidebar
st.sidebar.header("Search Restaurant")

##Adding a title
st.write(f"## Sentiment Analyser")

##Letting the user enter the business name and location
business_name = st.sidebar.text_input("Name (e.g. Chipotle)", "")
location = st.sidebar.text_input("Location (City, State Code)", "")
num_reviews = st.sidebar.number_input("Input number of reviews to diaply (upto 20)", min_value = 1, max_value = 20, step = 1)

#search button when clicked triggers a response
search_button = st.sidebar.button("Find locations")

##IF both business name and location are provided then we first get all results for said business and location
if search_button and business_name and location:
    results = search_businesses(business_name, location)

    #Adding only those results that have place_ids (which will be used to get the reviews for businesses)
    final_results = []
    for r in results:
        if 'place_ids' in r:
            final_results.append(r)
    # results = [r for r in results if "place_ids" in r]  # filter valid ones

    #Storing the current session state to be accessed later
    st.session_state["final_results"] = final_results

#Checking if the key 'final_results' exists in the session state and if its empty or not 
if "final_results" in st.session_state and st.session_state["final_results"]:
    display_results = st.session_state["final_results"]

    #Getting all locations in a list to create a dropdown
    location_options = []
    for r in display_results:
        location_options.append(f"{r['title']} — {r.get('neighborhoods', [])}")
        
    # [f"{r['title']} — {r.get('neighborhoods', [])}" for r in display_results]
    
    #Getting a dropdown from the options above and getting the index for the selected option
    selected_id = st.sidebar.selectbox("Choose a location:", range(len(location_options)),
                                        format_func=lambda i: location_options[i])
    
    #Getting top reviews for the selected place from the dropdown
    chosen_location = display_results[selected_id]["place_ids"][0]
    with st.spinner("Loading reviews...."): #Displaying this while loading reviews
        # num_reviews = st.text_input()
        #Getting the number of reviews asked by the user
        reviews = get_reviews(chosen_location, num_reviews=num_reviews)

        #Getting the ratings for all chosen reviews
        ratings = []
        for rev in reviews:
            if rev.get('rating') is not None:
                ratings.append(rev.get('rating'))
        # ratings = [rev.get('rating') for rev in reviews if rev.get('rating') is not None]

    ##Getting all reviews 
    rev_texts = []
    for r in reviews:
        rev_texts.append(r.get("comment", {}).get("text") or r.get("text", ""))
    # texts = [r.get("comment", {}).get("text") or r.get("text", "") for r in reviews]


    st.write(f"## Yelp Reviews for {location_options[selected_id]}")

    #If there are reviews, we first vectorize them and then send it into the svm model for prediction
    if rev_texts:
        X = vectorizer.transform(pd.Series(rev_texts))
        preds = svm_model.predict(X)

        df = pd.DataFrame({
            "Review": rev_texts,
            "Rating": ratings,
            "Sentiment": preds
        })
        # st.dataframe(df)

        ##Getting the average ratings for selected reviews
        if ratings:
            avg_rating = round(sum(ratings) / len(ratings), 2)
        else:
            avg_rating = "N/A"

        
        # Summary of the reviews
        counts = df["Sentiment"].value_counts().to_dict()
        st.subheader(f"Summary from top {num_reviews} reviews:")
        st.metric(f"Average Rating (Top {num_reviews}):", avg_rating)
        st.metric(":green[POSITIVE REVIEWS]", counts.get("positive", 0))
        st.metric(":red[NEGATIVE REVIEWS]", counts.get("negative", 0))
        st.metric("NEUTRAL REVIEWS", counts.get("neutral", 0))
        

        # #Wrapping the text in the dataframe 
        # styled_df = df.style.set_table_styles([{'selector': 'td', 'props': [('white-space', 'normal')]}]).hide(axis='index')

        # Function to color full row based on Sentiment
        def row_style(row):
            sentiment = row["Sentiment"]
            if sentiment == "positive":
                return ["background-color: #d4edda; color: green;"] * len(row)
            elif sentiment == "negative":
                return ["background-color: #f8d7da; color: red;"] * len(row)
            elif sentiment == "neutral":
                return ["background-color: #fefefe; color: gray;"] * len(row)
            else:
                return [""] * len(row)

        # Apply styling to the whole row
        styled_df = (
            df.style
            .apply(row_style, axis=1)
            .set_table_styles([{'selector': 'td', 'props': [('white-space', 'normal')]}])
            .hide(axis='index')
        )

        st.write(f"### Top {num_reviews} reviews")

        #Displaying the dataframe
        st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)

    else:
        st.warning("No reviews found.")
