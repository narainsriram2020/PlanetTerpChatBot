import streamlit as st
import google.generativeai as genai
import uuid
import datetime
import json
import random

# Basic setup
# Configure page layout with sidebar options
st.set_page_config(
    page_title="PlanetTerp Chatbot", 
    page_icon="🐢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for fixed sidebar elements
st.markdown("""
<style>
    [data-testid="stSidebar"] > div:first-child {
        height: 100vh;
        overflow: auto;
        padding-bottom: 180px;  /* Make space for the footer */
    }
    #umd-fact-container {
        position: fixed;
        bottom: 0;
        width: 18%;  /* Match sidebar width - may need adjustment */
        background-color: white;
        padding: 10px;
        border-top: 1px solid #ddd;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# Import all functions from planetterp_core
from planetterp_core import (
    load_model, get_courses, get_course, get_professor, get_course_grades,
    extract_course_ids, initialize_index, search, generate_response,
    generate_chat_name, get_greeting
)


# Initialize the model and state
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat" not in st.session_state:
    st.session_state.chat = genai.GenerativeModel('gemini-2.0-flash').start_chat(history=[])
if "context" not in st.session_state:
    st.session_state.context = {"courses": [], "professors": [], "grades": []}
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.course_ids = []
if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())
if "current_chat_name" not in st.session_state:
    st.session_state.current_chat_name = "New Chat"
if "first_visit" not in st.session_state:
    st.session_state.first_visit = True
if "umd_fact" not in st.session_state:
    st.session_state.umd_fact = None

# UMD Fun Facts
UMD_FACTS = [
    "UMD was founded in 1856 and is the flagship institution of the University System of Maryland.",
    "You can see the Washington Monument from the top of SECU Stadium",
    "Testudo, UMD's mascot, is a diamondback terrapin. Students rub its nose for good luck before exams!",
    
    "UMD has over 40,000 students from all 50 states and 118 countries.",
    "The University of Maryland's colors are red, white, black, and gold.",
    "People leave offerings to the Testudo statue in front of McKeldin Library. One time, things got out of hand and Testudo caught on fire.",
    "UMD has won national championships in men's basketball, women's basketball, men's lacrosse, women's lacrosse, field hockey, and football."
]

# Get a random UMD fact
def get_random_umd_fact():
    if "umd_fact" not in st.session_state or st.session_state.umd_fact is None:
        st.session_state.umd_fact = random.choice(UMD_FACTS)
    return st.session_state.umd_fact

# Add to session state initialization
if "umd_fact" not in st.session_state:
    st.session_state.umd_fact = None

# Cache the courses data
@st.cache_data(ttl=3600)
def cached_get_courses():
    return get_courses()

@st.cache_data(ttl=3600)
def cached_get_course(course_id):
    return get_course(course_id)

@st.cache_data(ttl=3600)
def cached_get_course_grades(course_id):
    return get_course_grades(course_id)

# Ensure the index is initialized
def ensure_index_initialized():
    if st.session_state.index is None:
        courses = cached_get_courses()
        
        # If courses is empty, display an error
        if not courses:
            st.sidebar.error("No courses retrieved. Please check the API.")
            return
            
        # Initialize the index
        st.session_state.index, st.session_state.course_ids = initialize_index(model, courses)

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
    st.session_state.context = {"courses": [], "professors": [], "grades": []}
    st.session_state.umd_fact = random.choice(UMD_FACTS)

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

# Basic UI
st.title("🐢 PlanetTerp Chatbot")
if st.session_state.first_visit:
    greeting = get_greeting()
    st.success(f"{greeting}! Welcome to PlanetTerp Chatbot. Ask me anything about UMD courses and professors.")
    st.session_state.first_visit = False

# Main chat area
for msg in st.session_state.messages:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"  # Change to any emoji or custom avatars
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Chat input
if query := st.chat_input("Ask about UMD courses..."):
    with st.spinner("🐢 Thinking..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        
        # Ensure index is initialized
        ensure_index_initialized()
        
        # First try to directly extract course IDs from the query
        direct_course_ids = extract_course_ids(query)
        
        # If no direct matches, use semantic search
        semantic_course_ids = [] if direct_course_ids else search(
            query, 
            st.session_state.index, 
            st.session_state.course_ids, 
            model
        )
        
        # Combine results, prioritizing direct matches
        all_course_ids = direct_course_ids + semantic_course_ids
        
        # Process the data
        data = {"courses": [], "professors": [], "grades": []}

        # Get course details
        for course_id in all_course_ids[:3]:  # Limit to top 3 results
            course = cached_get_course(course_id)
            if course:
                data["courses"].append(course)
                
                # Get grades for this course - now filtered and sorted by recency
                grades = cached_get_course_grades(course_id)
                if grades:
                    data["grades"].extend(grades[:5])  # Limit to 5 most recent grade entries
                
                # Get professors for this course
                for prof_name in course.get("professors", [])[:5]:  # Limit to 5 professors
                    prof = get_professor(prof_name)
                    if prof:
                        data["professors"].append(prof)
        
        # Update context
        st.session_state.context = data
        
        # Generate response
        response = generate_response(query, data, st.session_state.chat)
        
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

# Sidebar layout using custom CSS to ensure fixed positioning
st.markdown("""
<style>
    .sidebar-content {
        position: fixed;
        top: 0;
        bottom: 0;
        width: inherit;
        overflow: hidden;
        display: flex;
        flex-direction: column;
    }
    .sidebar-header {
        flex: 0 0 auto;
        padding-bottom: 1rem;
    }
    .sidebar-scroll {
        flex: 1 1 auto;
        overflow-y: auto;
        padding-right: 1rem;
        margin-right: -1rem;
    }
    .sidebar-footer {
        flex: 0 0 auto;
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for chat history
with st.sidebar:
    # Title section
    st.title("Chats")
    
    # Small fun fact at the top
    st.markdown(f"<small><i> Fun Fact: {get_random_umd_fact()}</i></small>", unsafe_allow_html=True)
    
    # New Chat button with icon
    if st.button("🔄 New Chat", key="new_chat_button", use_container_width=True):
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