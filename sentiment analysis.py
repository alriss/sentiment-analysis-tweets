import pandas as pd
from transformers import pipeline
import gradio as gr
import numpy

# Load the training and validation datasets
train_df = pd.read_csv('twitter_training.csv', names=['id','entity','sentiment','content'])
valid_df = pd.read_csv('twitter_validation.csv', names=['id','entity','sentiment','content'])

# Initialize sentiment-analysis pipeline with BERT
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to analyze sentiment of a given tweet
def analyze_sentiment(tweet):
    result = sentiment_pipeline(tweet)
    return result[0]['label']

# Function to fetch tweets based on entity and analyze their sentiment
def fetch_and_analyze(entity, dataset='train', count=10):
    if dataset == 'train':
        df = train_df
    else:
        df = valid_df
    # Filter tweets by entity and get the specified number of tweets
    tweets = df[df['entity'] == entity].head(count)
    # Analyze sentiment using the pre-trained model
    results = [(row['content'], analyze_sentiment(row['content'])) for _, row in tweets.iterrows()]
    return results

# Define the Gradio interface
interface = gr.Interface(
    fn=fetch_and_analyze,
    inputs=[
        gr.Dropdown(choices=list(train_df['entity'].unique()), value='', label="Entity"),
        gr.Radio(choices=["train", "valid"], label="Dataset"),
        gr.Slider(1, 100, step=1, value=10, label="Number of Tweets")
    ],
    outputs=gr.Dataframe(headers=["Tweet", "Predicted Sentiment"]),
    title="Twitter Sentiment Analysis",
    description="Enter an entity to analyze the sentiment of related tweets from the dataset."
)

# Launch the interface
interface.launch()
