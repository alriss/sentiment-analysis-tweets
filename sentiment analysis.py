import pandas as pd
from transformers import pipeline
from sklearn.cluster import KMeans
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# Load the training and validation datasets
train_df = pd.read_csv('twitter_training.csv', names=['id', 'entity', 'sentiment', 'content'])
valid_df = pd.read_csv('twitter_validation.csv', names=['id', 'entity', 'sentiment', 'content'])

# Initialize sentiment-analysis pipeline with BERT
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to analyze sentiment of a given tweet
def analyze_sentiment(tweet):
    result = sentiment_pipeline(tweet)
    return result[0]['score'] * ((result[0]['label'] == "POSITIVE") * 2 - 1)

# Function to fetch tweets based on entity
def fetch_tweets(entity, dataset='train', count=10):
    if dataset == 'train':
        df = train_df
    else:
        df = valid_df
    # Filter tweets by entity and get the specified number of tweets
    tweets = df[df['entity'] == entity].head(count)
    return tweets['content'].tolist()

# Function to determine the optimal number of clusters using the elbow method
def determine_optimal_clusters(data):
    sse = []
    k_range = range(1, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        sse.append(kmeans.inertia_)
    optimal_k = k_range[np.diff(sse).argmin() + 1]  # +1 due to zero-indexing
    return optimal_k

# Function to perform clustering and generate an interactive scatterplot
def cluster_and_plot(tweets):
    # Compute TF-IDF vectors for the tweets
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())

    # Determine the optimal number of clusters
    optimal_k = determine_optimal_clusters(reduced_data)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(reduced_data)
    labels = kmeans.labels_

    # Create an interactive scatter plot using Plotly
    fig = px.scatter(
        x=reduced_data[:, 0], 
        y=reduced_data[:, 1], 
        color=labels, 
        hover_data={'Tweet': tweets}
    )
    fig.update_layout(
        title='Tweet Clustering Based on TF-IDF and PCA',
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        coloraxis_colorbar=dict(title='Cluster Label'),
        template='plotly_dark'
    )

    # Save the plot to a file and return the filename
    plot_filename = 'scatter_plot.html'
    fig.write_html(plot_filename)

    return plot_filename

# Define the Gradio interface
def fetch_analyze_and_cluster(entity, dataset='train', count=10):
    tweets = fetch_tweets(entity, dataset, count)
    plot_filename = cluster_and_plot(tweets)
    return plot_filename

# Define the Gradio interface
interface = gr.Interface(
    fn=fetch_analyze_and_cluster,
    inputs=[
        gr.Dropdown(choices=list(train_df['entity'].unique()), value='Borderlands', label="Entity"),
        gr.Radio(choices=["train", "valid"], label="Dataset"),
        gr.Slider(1, 100, step=1, value=10, label="Number of Tweets")
    ],
    outputs=gr.HTML(label="Cluster Scatter Plot"),
    title="Twitter Sentiment Analysis",
    description="Enter an entity to analyze the sentiment of related tweets from the dataset."
)

# Launch the interface
interface.launch()
