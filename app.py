import streamlit as st
import google.generativeai as genai
import uuid
import datetime
import json
import random

from planetterp_core import (
    load_model, get_courses, get_course, get_professor, get_course_grades,
    extract_course_ids, initialize_index, search, generate_response,
    generate_chat_name, get_greeting
)

def get_random_umd_fact():
    umd_facts = [
        "UMD's mascot Testudo is a diamondback terrapin, Maryland's state reptile.",
        "McKeldin Mall is the largest academic mall in the country.",
        "UMD has the oldest continuously operating airport in the world - College Park Airport.",
        "UMD's school colors (red, white, black, and gold) come from the Maryland state flag.",
        "UMD's campus spans over 1,300 acres.",
        "The 'M Circle' flowerbed is 57 feet in diameter.",
        "UMD is one of only 62 members of the Association of American Universities.",
        "The Xfinity Center can hold over 17,000 fans for basketball games.",
        "UMD's campus has over 8,000 trees of 400+ species.",
        "Morrill Hall is UMD's oldest academic building, completed in 1898.",
        "The Clarice Smith Performing Arts Center covers 318,000 square feet.",
        "The fear of turtles is called chelonaphobia.",
        "Testudo statues around campus are considered good luck, especially during finals week.",
    ]
    return random.choice(umd_facts)

st.set_page_config(page_title="PlanetTerp Chatbot", page_icon="🐢")

# Initialize the model and state
@st.cache_resource
def get_model():
    try:
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        return load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = get_model()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat" not in st.session_state:
    try:
        st.session_state.chat = genai.GenerativeModel(
            'gemini-2.0-flash',
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024,
            },
        ).start_chat(history=[])
    except Exception as e:
        st.error(f"Error initializing chat: {str(e)}")
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
if "current_fact" not in st.session_state:
    st.session_state.current_fact = get_random_umd_fact()

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
        if not courses:
            st.sidebar.error("No courses retrieved. Please check the API.")
            return
        st.session_state.index, st.session_state.course_ids = initialize_index(model, courses)

# Function to save the current chat
def save_current_chat():
    if st.session_state.messages:
        if st.session_state.current_chat_name == "New Chat" and st.session_state.messages:
            first_user_message = next((msg["content"] for msg in st.session_state.messages if msg["role"] == "user"), None)
            if first_user_message:
                st.session_state.current_chat_name = generate_chat_name(first_user_message)
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
    st.session_state.chat = genai.GenerativeModel(
        'gemini-2.0-flash',
        generation_config={
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        },
    ).start_chat(history=[])
    st.session_state.context = {"courses": [], "professors": [], "grades": []}
    
    # Update the fun fact only when a new chat is created
    st.session_state.current_fact = get_random_umd_fact()

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
        
        st.session_state.chat = genai.GenerativeModel(
            'gemini-2.0-flash',
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024,
            },
        ).start_chat(history=chat_history)

# Basic UI
st.title("🐢 PlanetTerp Chatbot")
if st.session_state.first_visit:
    greeting = get_greeting()
    st.markdown(f"*{greeting}!* Welcome to PlanetTerp Chatbot. Ask me anything about UMD courses and professors.")
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
            st.rerun()

with st.sidebar:
    # Fun Fact Section
    st.markdown("### Fun Fact")
    with st.container(border=False):
        st.markdown(f"*{st.session_state.current_fact}*")

    st.divider()
    
    # New Chat Button Section
    st.markdown("### Chat")
    if st.button("+ New Chat", key="new_chat_button", use_container_width=True):
        start_new_chat()
        st.rerun()
    
    st.divider()

    # How to Use Section
    st.markdown("### How to Use This Chatbot")
    st.markdown("""
    **🔍 Ask about courses:**
    - Search by course ID (e.g., "Tell me about CMSC131")
    - Get professor ratings for specific courses
    - Find prerequisites and course descriptions
    
    **👨‍🏫 Learn about professors:**
    - Ask about teaching styles and ratings
    - Compare professors for the same course
    - Find what courses a professor teaches
    
    **💡 Tips:**
    - Be specific with course IDs for best results
    - Try asking for recommendations based on interests
    - Use natural language for your questions
    """)
    
    st.divider()
    
    # Quick Links for Students
    st.markdown("### Quick UMD Links")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("[📝 ELMS](https://elms.umd.edu)")
        st.markdown("[📅 Testudo](https://testudo.umd.edu)")
    with col2:
        st.markdown("[📚 Classes (SOC)](https://app.testudo.umd.edu/soc/)")
        st.markdown("[🍽️ Dining](https://dining.umd.edu)")
    
    st.divider()
    
    # Academic Calendar Highlights
    st.markdown("### Important Dates")
    st.markdown("🏝️ **Spring Break**: March 16-23")
    st.markdown("🗓️ **Registration**: April 1-15, 2025")
    st.markdown("📚 **Finals Exams**: May 15-21, 2025")
    st.markdown("🎓 **Commencement**: May 21, 2025")
    
    # Version Info
    st.caption("v1.0.0 | Updated March 2025")