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
import uuid
import datetime

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

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat" not in st.session_state:
    st.session_state.chat = genai.GenerativeModel('gemini-2.0-flash').start_chat(history=[])
if "context" not in st.session_state:
    st.session_state.context = {"courses": [], "professors": []}
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.course_ids = []
if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())
if "current_chat_name" not in st.session_state:
    st.session_state.current_chat_name = "New Chat"

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
    response = requests.get(f"{PLANETTERP_BASE_URL}/course", params={"name": course_id})
    if response.status_code == 200:
        return response.json()
    st.sidebar.warning(f"Course not found: {course_id}, Status: {response.status_code}")
    return None

def get_professor(name):
    response = requests.get(f"{PLANETTERP_BASE_URL}/professor", 
                           params={"name": name, "reviews": "true"})
    if response.status_code == 200:
        professor_data = response.json()
        # Sort reviews by recency
        return sort_professor_reviews(professor_data)
    return None

# Add this function to sort and filter grades by recency
def filter_recent_grades(grades, years=4):
    # Get current year
    from datetime import datetime
    current_year = datetime.now().year
    
    # Filter grades from the last 'years' years
    recent_grades = [g for g in grades if g.get('semester') and 
                    int(g.get('semester', '000000')[:4]) >= (current_year - years)]
    
    # Sort by semester in descending order (most recent first)
    recent_grades.sort(key=lambda x: x.get('semester', '000000'), reverse=True)
    
    return recent_grades

def sort_professor_reviews(professor_data):
    if professor_data and 'reviews' in professor_data and professor_data['reviews']:
        # Sort reviews by date in descending order
        professor_data['reviews'].sort(key=lambda x: x.get('created', ''), reverse=True)
    return professor_data

# Modify the get_course_grades function to filter and sort by recency
@st.cache_data(ttl=3600)
def get_course_grades(course_id):
    response = requests.get(f"{PLANETTERP_BASE_URL}/grades", params={"course": course_id})
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, dict) and "error" in data:
            return []
        # Filter and sort grades by recency
        return filter_recent_grades(data)
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

def get_rating_emoji(rating):
    """Convert numerical rating to emoji for visual representation"""
    if not rating or rating < 0:
        return " ❓"  # Unknown rating
    elif rating >= 4.5:
        return " 🌟"  # Excellent
    elif rating >= 4.0:
        return " ⭐"  # Very Good
    elif rating >= 3.0:
        return " ✅"  # Good/Average
    elif rating >= 2.0:
        return " ⚠️"  # Below Average
    else:
        return " ❗"  # Poor

# Generate response
def generate_response(query, data):
    system = """
    You are a UMD assistant using PlanetTerp data. Be concise but helpful about courses and professors.
    Focus on the most recent grades, ratings, and recommendations from the past 3-4 years (2021-2025).
    Explicitly mention the recency of the data (e.g., "According to Spring 2024 data...").
    Remember context from previous questions. Keep in mind what class or professors you have already suggested during the converation and use that as context. 
    Do not bring up random data out of nowhere continue conversation on whatever data is being currently discussed. 
    If you don't have information about a specific course or professor, simply say so and explain that
    they should visit the PlanetTerp website to get more info, but this should be the last resort option.
    Sound very laid back and chill like your are another student.
    When a student asks about what professor they should take for a certain course give them your personal evaluation of the best professor.
    When mentioning professors, always include their rating emoji beside their name using this format:
    Professor Name [emoji] - where emoji indicates their average rating.
    """
    
    for prof in data["professors"]:
        if "average_rating" in prof:
            prof["rating_emoji"] = get_rating_emoji(prof["average_rating"])
        else:
            prof["rating_emoji"] = "❓"
    
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

# Function to generate a chat name based on the first user query
def generate_chat_name(query):
    # If there are course IDs in the query, use them in the name
    course_ids = extract_course_ids(query)
    if course_ids:
        return f"{', '.join(course_ids)} Question"
    
    # Otherwise, use the first few words of the query
    words = query.split()
    if len(words) <= 4:
        return query
    else:
        return ' '.join(words[:4]) + "..."

# Function to save the current chat
def save_current_chat():
    # Only save if there are messages
    if st.session_state.messages:
        # Generate a chat name if this is the first message
        if st.session_state.current_chat_name == "New Chat" and st.session_state.messages:
            first_user_message = next((msg["content"] for msg in st.session_state.messages if msg["role"] == "user"), None)
            if first_user_message:
                st.session_state.current_chat_name = generate_chat_name(first_user_message)
        
        # Save the chat with its current ID and name
        st.session_state.saved_chats[st.session_state.current_chat_id] = {
            "name": st.session_state.current_chat_name,
            "messages": st.session_state.messages.copy(),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        }

# Function to start a new chat
def start_new_chat():
    # Save the current chat first
    save_current_chat()
    
    # Create a new chat
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.current_chat_name = "New Chat"
    st.session_state.messages = []
    st.session_state.chat = genai.GenerativeModel('gemini-2.0-flash').start_chat(history=[])
    st.session_state.context = {"courses": [], "professors": []}

# Function to load a saved chat
def load_chat(chat_id):
    if chat_id in st.session_state.saved_chats:
        # Save current chat
        save_current_chat()
        
        # Load the selected chat
        chat_data = st.session_state.saved_chats[chat_id]
        st.session_state.current_chat_id = chat_id
        st.session_state.current_chat_name = chat_data["name"]
        st.session_state.messages = chat_data["messages"].copy()
        
        # Re-initialize the chat model with the history
        chat_history = []
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "model"
            chat_history.append({"role": role, "parts": [msg["content"]]})
        
        st.session_state.chat = genai.GenerativeModel('gemini-2.0-flash').start_chat(history=chat_history)

# Add this to your session state initialization
if "first_visit" not in st.session_state:
    st.session_state.first_visit = True

# Add this function to determine time of day greeting
def get_greeting():
    hour = datetime.datetime.now().hour
    if hour > 5 and hour < 11:
        return "Good morning"
    elif hour > 12 and hour < 15:
        return "Good afternoon"
    else:
        return "Good evening"


# Basic UI
st.title("🐢 PlanetTerp Chatbot")
if st.session_state.first_visit:
    greeting = get_greeting()
    st.success(f"{greeting}! Welcome to PlanetTerp Chatbot. Ask me anything about UMD courses and professors.")
    st.session_state.first_visit = False

# Main chat area
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if query := st.chat_input("Ask about UMD courses..."):
    with st.spinner("🐢 Thinking..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        
        # First try to directly extract course IDs from the query
        direct_course_ids = extract_course_ids(query)
        
        # If no direct matches, use semantic search
        semantic_course_ids = [] if direct_course_ids else search(query)
        
        # Combine results, prioritizing direct matches
        all_course_ids = direct_course_ids + semantic_course_ids
        
        # Process the data
        data = {"courses": [], "professors": [], "grades": []}

        # Get course details
        for course_id in all_course_ids[:3]:  # Limit to top 3 results
            course = get_course(course_id)
            if course:
                data["courses"].append(course)
                
                # Get grades for this course - now filtered and sorted by recency
                grades = get_course_grades(course_id)
                if grades:
                    data["grades"].extend(grades[:5])  # Limit to 5 most recent grade entries
                
                # Get professors for this course
                for prof_name in course.get("professors", [])[:5]:  # Limit to 3 professors
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
        
        # If this is the first message in a new chat, generate a name
        if st.session_state.current_chat_name == "New Chat":
            st.session_state.current_chat_name = generate_chat_name(query)
            # Save the chat with the new name
            save_current_chat()
            # Force a rerun to update the sidebar
            st.experimental_rerun()

# Sidebar for chat history
with st.sidebar:
    st.title("Chats")
    
    # New Chat button
    if st.button("+ New Chat", key="new_chat_button"):
        start_new_chat()
        st.experimental_rerun()
    
    st.divider()
    
    # Display saved chats
    if st.session_state.saved_chats:
        st.subheader("Chat History")
        
        # Sort chats by timestamp (most recent first)
        sorted_chats = sorted(
            st.session_state.saved_chats.items(),
            key=lambda x: x[1]["timestamp"],
            reverse=True
        )
        
        # Display each chat as a button
        for chat_id, chat_data in sorted_chats:
            # Highlight the current chat
            button_style = "primary" if chat_id == st.session_state.current_chat_id else "secondary"
            
            # Create a unique key for each button
            button_key = f"chat_{chat_id}"
            
            if st.button(
                chat_data["name"], 
                key=button_key,
                type=button_style,
                use_container_width=True
            ):
                load_chat(chat_id)
                st.experimental_rerun()