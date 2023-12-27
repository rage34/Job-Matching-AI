import streamlit as st
import requests
from PyPDF2 import PdfReader
from rake_nltk import Rake
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# Download NLTK stopwords data
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def fetch_job_description(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching job description: {e}")
        return None

def read_pdf_resume(uploaded_file):
    try:
        # Check file size
        max_file_size = 2 * 1024 * 1024  # 2MB in bytes
        if uploaded_file.size > max_file_size:
            st.error("File size exceeds the maximum allowed (2MB). Please upload a smaller file.")
            return None

        pdf_reader = PdfReader(uploaded_file)
        resume_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            resume_text += page.extract_text()
        return resume_text
    except Exception as e:
        st.error(f"Error reading PDF resume: {e}")
        return None

def calculate_similarity_score(job_description, resume_text):
    # Custom Stopwords
    custom_stopwords = set(['e', '15', 'com', '30', '90', '91', 'end'])

    # Rake Keywords
    r = Rake(stopwords=list(stopwords.words('english')) + list(custom_stopwords))
    r.extract_keywords_from_text(job_description)
    job_keywords = set(r.get_ranked_phrases())

    r.extract_keywords_from_text(resume_text)
    resume_keywords = set(r.get_ranked_phrases())

    # Cosine Similarity
    vectorizer = CountVectorizer(stop_words=list(stopwords.words('english')) + list(custom_stopwords)).fit_transform([job_description, resume_text])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

    # Calculate compatibility score
    compatibility_score = (len(job_keywords.intersection(resume_keywords)) / len(job_keywords)) * cosine_sim

    return compatibility_score*100000, job_keywords, resume_keywords

# Streamlit UI
st.markdown("<h1 style='text-align: center; font-size: 2.5em; font-weight: bold;'>Job Matching AI</h1>", unsafe_allow_html=True)

# Input job description URL
job_description_url = st.text_input("Enter Job Description URL:")
if job_description_url:
    job_description = fetch_job_description(job_description_url)

# Upload resume PDF
resume_upload = st.file_uploader("Upload Resume PDF (PDF format only, max 2MB)", type=["pdf"])
if resume_upload:
    resume_text = read_pdf_resume(resume_upload)

# Button to trigger calculation
if st.button("Calculate") and job_description_url and resume_text:
    # Calculate compatibility score if both job description and resume are available
    compatibility_score, job_keywords, resume_keywords = calculate_similarity_score(job_description, resume_text)

    # Display compatibility score
    st.subheader("Compatibility Score:")
    st.write(f"{compatibility_score:.2%}")

    # Display common keywords
    common_keywords = job_keywords.intersection(resume_keywords)
    st.subheader("Common Keywords:")
    for keyword in common_keywords:
        st.write("-", keyword)
