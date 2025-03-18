import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import json
import os
from dotenv import load_dotenv
import re

# Basic setup
load_dotenv()
st.set_page_config(page_title="PlanetTerp Chatbot", page_icon="🐢")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
PLANETTERP_BASE_URL = "https://api.planetterp.com/v1"

# Initialize the model and state
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat" not in st.session_state:
    st.session_state.chat = genai.GenerativeModel('gemini-1.5-pro-latest').start_chat(history=[])
if "context" not in st.session_state:
    st.session_state.context = {"courses": [], "professors": []}
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.course_ids = []

# Core API functions
@st.cache_data(ttl=3600)
def get_courses():
    response = requests.get(f"{PLANETTERP_BASE_URL}/courses")
    if response.status_code == 200:
        return response.json()
    st.sidebar.error(f"Failed to fetch courses: {response.status_code}")
    return []

@st.cache_data(ttl=3600)
def get_course(course_id):
    # Debug log
    st.sidebar.write(f"Fetching course: {course_id}")
    response = requests.get(f"{PLANETTERP_BASE_URL}/course", params={"name": course_id})
    if response.status_code == 200:
        return response.json()
    st.sidebar.warning(f"Course not found: {course_id}, Status: {response.status_code}")
    return None

@st.cache_data(ttl=3600)
def get_professor(name):
    response = requests.get(f"{PLANETTERP_BASE_URL}/professor", params={"name": name})
    if response.status_code == 200:
        return response.json()
    return None

@st.cache_data(ttl=3600)
def get_course_grades(course_id):
    response = requests.get(f"{PLANETTERP_BASE_URL}/grades", params={"course": course_id})
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, dict) and "error" in data:
            return []
        return data
    return []

# Extract course IDs from query
def extract_course_ids(query):
    # Look for standard course patterns like CMSC330, MATH140, etc.
    pattern = r'\b[A-Z]{4}\d{3}[A-Z]?\b'
    matches = re.findall(pattern, query.upper())
    return matches

# Semantic search setup
def initialize_index():
    if st.session_state.index is None:
        courses = get_courses()
        
        # If courses is empty, display an error
        if not courses:
            st.sidebar.error("No courses retrieved. Please check the API.")
            return
            
        # Create descriptions
        texts = []
        ids = []
        for course in courses:
            course_id = course.get('course_id') or course.get('name')
            title = course.get('title', '')
            desc = course.get('description', '')
            texts.append(f"{course_id}: {title}. {desc}")
            ids.append(course_id)
        
        # Create embeddings
        embeddings = model.encode(texts)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype('float32'))
        
        st.session_state.index = index
        st.session_state.course_ids = ids
        st.sidebar.success(f"Index created with {len(ids)} courses")

def search(query, k=5):
    if st.session_state.index is None:
        initialize_index()
        if st.session_state.index is None:  # If still None after initialization
            return []
    
    query_embedding = model.encode([query])
    _, indices = st.session_state.index.search(
        np.array(query_embedding).astype('float32'), k
    )
    
    results = []
    for idx in indices[0]:
        if idx < len(st.session_state.course_ids):
            course_id = st.session_state.course_ids[idx]
            if course_id not in results:  # Avoid duplicates
                results.append(course_id)
    
    return results

# Generate response
def generate_response(query, data):
    system = """
    You are a UMD assistant using PlanetTerp data. Be concise but helpful about courses and professors.
    Focus on grades, ratings, and recommendations. Remember context from previous questions.
    If you don't have information about a specific course or professor, simply say so and explain that
    you're limited to the data available from PlanetTerp API.
    """
    
    context = {
        "courses": data["courses"],
        "professors": data["professors"],
        "grades": data["grades"]
    }
    
    prompt = f"""
    PlanetTerp Data: {json.dumps(context, indent=2)}
    Question: {query}
    """
    
    response = st.session_state.chat.send_message(system + prompt)
    return response.text

# Basic UI
st.title("🐢 PlanetTerp Semantic Search")

# Display initialization status
with st.sidebar:
    st.title("App Status")
    if st.session_state.index is None:
        st.info("Initializing course index...")
        initialize_index()
    else:
        st.success(f"Index ready with {len(st.session_state.course_ids)} courses")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if query := st.chat_input("Ask about UMD courses..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    # First try to directly extract course IDs from the query
    direct_course_ids = extract_course_ids(query)
    
    # If no direct matches, use semantic search
    semantic_course_ids = [] if direct_course_ids else search(query)
    
    # Combine results, prioritizing direct matches
    all_course_ids = direct_course_ids + semantic_course_ids
    
    # Debug info
    with st.sidebar:
        st.write("Query:", query)
        st.write("Direct matches:", ", ".join(direct_course_ids) if direct_course_ids else "None")
        st.write("Semantic matches:", ", ".join(semantic_course_ids) if semantic_course_ids else "None")
    
    data = {"courses": [], "professors": [], "grades": []}
    
    # Get course details
    for course_id in all_course_ids[:3]:  # Limit to top 3 results
        course = get_course(course_id)
        if course:
            data["courses"].append(course)
            
            # Get grades for this course
            grades = get_course_grades(course_id)
            if grades:
                data["grades"].extend(grades[:5])  # Limit to 5 grade entries
            
            # Get professors for this course
            for prof_name in course.get("professors", [])[:3]:  # Limit to 3 professors
                prof = get_professor(prof_name)
                if prof:
                    data["professors"].append(prof)
    
    # Update context
    st.session_state.context = data
    
    # Generate response
    response = generate_response(query, data)
    
    # Display response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

# Simple sidebar for previous chats
with st.sidebar:
    st.title("Chats")
    if st.button("New Chat"):
        st.session_state.messages = []
        st.session_state.chat = genai.GenerativeModel('gemini-1.5-pro-latest').start_chat()
        st.session_state.context = {"courses": [], "professors": []}
        st.experimental_rerun()
    
    # Direct course lookup for testing
    st.title("Debug Tools")
    with st.form("course_lookup"):
        test_course = st.text_input("Test Course ID", "CMSC330")
        submit = st.form_submit_button("Test Lookup")
        if submit:
            result = get_course(test_course)
            if result:
                st.success(f"Found: {result.get('name')} - {result.get('title')}")
                st.write(result)
            else:
                st.error(f"Course {test_course} not found")