import googleapiclient.discovery
import pandas as pd
import pycountry
from langdetect import detect, LangDetectException
import streamlit as st
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re

X_comments = pd.read_pickle('X_comments.pkl') 

stop_words = stopwords.words('english')
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
snowball_stemmer = SnowballStemmer(language = 'english')
lzr = WordNetLemmatizer()
max_sequence_length = 100
tokenizer = Tokenizer(num_words=45000, lower=True)
tokenizer.fit_on_texts(X_comments)
X_comments = tokenizer.texts_to_sequences(X_comments)
X_comments = pad_sequences(X_comments, maxlen=max_sequence_length)

with open('LSTM_Model', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = nltk.word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]

    lemmatized_tokens = [lzr.lemmatize(token) for token in tokens]

    preprocessed_text = ' '.join(lemmatized_tokens)
    
    return preprocessed_text


def get_polarity(text_comment):
    preprocessed_comment = preprocess_text(text_comment)  
    numeric_comment = tokenizer.texts_to_sequences([preprocessed_comment])
    numeric_comment = pad_sequences(numeric_comment, maxlen=max_sequence_length)

    predictions = loaded_model.predict(numeric_comment)

    sentiment_probabilities = predictions[1][0]

    highest_probability_index = sentiment_probabilities.argmax()

    compound_score = predictions[0][0]

    return compound_score, highest_probability_index

def det_lang(language):
    """ Function to detect language
    Args:
        Language column from the dataframe
    Returns:
        Detected Language or Other
    """
    try:
        lang = detect(language)
    except LangDetectException:
        lang = 'Other'
    return lang


def parse_video(url) -> pd.DataFrame:
    """
    Args:
      url: URL Of the video to be parsed
    Returns:
      Dataframe with the processed and cleaned values
    """

    # Get the video_id from the url
    video_id = url.split('?v=')[-1]

    # creating youtube resource object
    youtube = googleapiclient.discovery.build(
        'youtube', 'v3', developerKey="AIzaSyDVTPDPfewYqx5nt9bw5Mrkr9JQp_cpm64")

    # retrieve youtube video results
    video_response = youtube.commentThreads().list(
        part='snippet',
        maxResults=100,
        order='relevance',
        videoId=video_id
    ).execute()

    # empty list for storing reply
    comments = []

    # extracting required info from each result object
    for item in video_response['items']:

        # Extracting comments
        comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
        # Extracting author
        author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
        # Extracting published time
        published_at = item['snippet']['topLevelComment']['snippet']['publishedAt']
        # Extracting likes
        like_count = item['snippet']['topLevelComment']['snippet']['likeCount']
        # Extracting total replies to the comment
        reply_count = item['snippet']['totalReplyCount']

        comments.append(
            [author, comment, published_at, like_count, reply_count])

    df_transform = pd.DataFrame({'Author': [i[0] for i in comments],
                                 'Comment': [i[1] for i in comments],
                                 'Timestamp': [i[2] for i in comments],
                                 'Likes': [i[3] for i in comments],
                                 'TotalReplies': [i[4] for i in comments]})

    # Remove extra spaces and make them lower case. Replace special emojis
    df_transform['Comment'] = df_transform['Comment'].apply(lambda x: x.strip().lower().
                                                            replace('xd', '').replace('<3', ''))

    # Detect the languages of the comments
    df_transform['Language'] = df_transform['Comment'].apply(det_lang)

    # Convert ISO country codes to Languages
    df_transform['Language'] = df_transform['Language'].apply(
        lambda x: pycountry.languages.get(alpha_2=x).name if (x) != 'Other' else 'Not-Detected')

    # Dropping Not detected languages
    df_transform.drop(
        df_transform[df_transform['Language'] == 'Not-Detected'].index, inplace=True)

    df_transform['Polarity'], df_transform['Sentiment_Type'] = zip(*df_transform.apply(
        lambda x: (get_polarity(x['Comment']) if x['Language'] == 'English' else ('', '')), axis=1))

    # Replace sentiment types based on the given mapping
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    df_transform['Sentiment_Type'] = df_transform['Sentiment_Type'].map(sentiment_mapping)

    # Change the Timestamp
    df_transform['Timestamp'] = pd.to_datetime(
        df_transform['Timestamp']).dt.strftime('%Y-%m-%d %r')

    return df_transform

def youtube_metrics(url) -> list:
    """ Function to get views, likes and comment counts
    Args:
        URL: url of the youtube video
    Returns:
        List containing views, likes and comment counts
    """
    # Get the video_id from the url
    video_id = url.split('?v=')[-1]
    # creating youtube resource object
    youtube = googleapiclient.discovery.build(
        'youtube', 'v3', developerKey="AIzaSyDVTPDPfewYqx5nt9bw5Mrkr9JQp_cpm64")
    statistics_request = youtube.videos().list(
        part="statistics",
        id=video_id
    ).execute()

    metrics = []

    # extracting required info from each result object
    for item in statistics_request['items']:
        # Extracting views
        metrics.append(item['statistics']['viewCount'])
        # Extracting likes
        metrics.append(item['statistics']['likeCount'])
        # Extracting Comments
        metrics.append(item['statistics']['commentCount'])

    return metrics


if __name__ == "__main__":
    df_main = parse_video('https://www.youtube.com/watch?v=dQw4w9WgXcQ')
    df_yt = youtube_metrics('https://www.youtube.com/watch?v=dQw4w9WgXcQ')
    print(df_main.head())
    print(df_yt)