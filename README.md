# 🎬 The Movie Recommendation Engine

An intelligent movie recommendation system that helps you discover films tailored to your taste.

## 🌟 Features

- **Content-Based Filtering**: Analyzes movie features using keywords from the overview
- **TF-IDF Vectorization**: Advanced text analysis for accurate similarity scoring
- **Interactive UI**: Clean, modern interface built with Streamlit
- **Real-time Recommendations**: Get 10 similar movies instantly
- **Large Movie Database**: Thousands of films to explore

## 🛠️ Tech Stack

- **Python 3.12**
- **Streamlit** - Web interface
- **scikit-learn** - Machine learning algorithms
- **Pandas** - Data manipulation
- **joblib** - Model serialization

## 🚀 How It Works

1. Select your favorite movie from the dropdown
2. Click "Get Recommendations"
3. Receive 10 personalized movie suggestions based on similarity scoring

The system uses sigmoid kernel on TF-IDF vectors to find movies with similar content characteristics.

## 📊 Model Details

- **Algorithm**: Content-Based Filtering
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Similarity Metric**: Sigmoid Kernel

## 🎯 Use Cases

- Discover new movies similar to your favorites
- Find films in specific genres or styles
- Explore cinema beyond mainstream recommendations
