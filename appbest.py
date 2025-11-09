import pandas as pd
import joblib
import streamlit as st

st.set_page_config(
    page_title='The Movie Recommendation Engine',
    page_icon='🎬',
    layout='centered',
    initial_sidebar_state='collapsed'
)

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path + '/movie_data_for_app.csv')
    dataframe = pd.read_csv(file_path + '/movie_dataframe_for_app.csv')
    return data, dataframe

@st.cache_resource
def load_models(file_path):
    sig = joblib.load(file_path + '/sigmoid_kernel.pkl')
    tfv = joblib.load(file_path + '/tfidf_vectorizer.pkl')
    return sig, tfv

data, dataframe = load_data('C:/dumped_movie_recc_obj')
sig, tfv = load_models('C:/dumped_movie_recc_obj')

def give_recommendations(movie_title, model, data, dataframe):
    """Tell us what you love, we'll find your next obsession!"""
    indices = pd.Series(data=data.index, index=data['original_title'])
    idx = indices[movie_title]
    model_scores = list(enumerate(model[idx]))
    model_scores_sorted = sorted(model_scores, key=lambda x: x[1], reverse=True)
    model_scores_10 = model_scores_sorted[1:11]
    movie_indices_10 = [i[0] for i in model_scores_10]
    
    recommendations = []
    for i in movie_indices_10:
        movie_name = dataframe.iloc[i]['original_title']
        recommendations.append(movie_name)
    
    return recommendations

st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem 1rem;
    }
    
    /* Title styling */
    .title-container {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #666;
        font-weight: 400;
    }
    
    /* Select box styling - BIGGER & MORE VISIBLE */
    .stSelectbox label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff !important;
        margin-bottom: 0.75rem !important;
    }
    
    /* Main selectbox container */
    .stSelectbox div[data-baseweb="select"] {
        min-height: 60px !important;
    }
    
    /* The selected value display */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        font-size: 1.15rem !important;
        font-weight: 500 !important;
        padding: 1rem 1.25rem !important;
        min-height: 60px !important;
        line-height: 1.5 !important;
        display: flex !important;
        align-items: center !important;
        border-radius: 10px !important;
        border: 2px solid #4a5568 !important;
    }
    
    /* Text inside selectbox */
    .stSelectbox div[data-baseweb="select"] span {
        color: #ffffff !important;
        font-size: 1.15rem !important;
        line-height: 1.5 !important;
    }
    
    /* Dropdown icon */
    .stSelectbox svg {
        fill: #ffffff !important;
        width: 24px !important;
        height: 24px !important;
    }
    
    /* Hover state */
    .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: #667eea !important;
        box-shadow: 0 0 0 1px #667eea !important;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Recommendation card */
    .rec-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 2rem 0 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .rec-header h3 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .rec-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
        font-size: 1rem;
    }
    
    /* Movie item card */
    .movie-card {
        background: white;
        border: 2px solid #e8e8e8;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        display: flex;
        align-items: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .movie-card:hover {
        border-color: #667eea;
        transform: translateX(10px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .movie-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1rem;
        margin-right: 1rem;
        flex-shrink: 0;
    }
    
    .movie-title {
        font-size: 1.15rem;
        color: #2c3e50;
        font-weight: 500;
        flex-grow: 1;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 2px solid #e8e8e8;
    }
    
    .footer-title {
        font-weight: 600;
        color: #2c3e50;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .footer-text {
        color: #7f8c8d;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .spacer {
        height: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-container">
        <h1 class="main-title">🎬 The Movie Recommendation Engine</h1>
        <p class="subtitle">Tell us what you love, we'll find your next obsession!</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("### 🔍 Choose Your Favorite Movie")

movie_list = sorted([movie for movie in data['original_title'].unique() if movie and str(movie).strip()])

if 'selected_movie' not in st.session_state:
   st.session_state.selected_movie = ""

selected_movie = st.selectbox(
    'Type to search or click to browse',
    options=movie_list,
    index=movie_list.index(st.session_state.selected_movie) if st.session_state.selected_movie in movie_list else 0
)

st.session_state.selected_movie = selected_movie

st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    get_recommendations = st.button('✨ Get Recommendations', type='primary')

if get_recommendations and selected_movie:
    with st.spinner('🎯 Finding your perfect matches...'):
        try:
            recommendations = give_recommendations(selected_movie, sig, data, dataframe)
            
            st.markdown(f"""
                <div class="rec-header">
                    <h3>🎯 Movies Similar to: {selected_movie}</h3>
                    <p>Based on content analysis and similarity scoring</p>
                </div>
            """, unsafe_allow_html=True)
            
            for idx, movie in enumerate(recommendations, 1):
                st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-number">{idx}</div>
                        <div class="movie-title">{movie}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.success('🎉 Enjoy your movie marathon!')
            
        except Exception as e:
            st.error(f'❌ Error: {str(e)}')
            st.info('Please try selecting a different movie.')

st.markdown("""
    <div class="footer">
        <p class="footer-title">💡 How It Works</p>
        <p class="footer-text">
            This app uses <strong>Content-Based Filtering</strong> with TF-IDF vectorization<br>
            to analyze movie features and find the most similar movies to your selection.
        </p>
    </div>
""", unsafe_allow_html=True)