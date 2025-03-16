import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import requests
import os
import json
import pandas as pd
import plotly.express as px

# Configuration
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Please set your GEMINI_API_KEY in the .env file")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
PLANETTERP_BASE_URL = "https://api.planetterp.com/v1"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat" not in st.session_state:
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    st.session_state.chat = model.start_chat(history=[])

@st.cache_data(ttl=3600, show_spinner=False)
def search_courses(query):
    """Search for courses by name or course ID"""
    try:
        response = requests.get(f"{PLANETTERP_BASE_URL}/search", 
                               params={"query": query, "type": "course"}, 
                               timeout=10)
        return response.json() if response.status_code == 200 else []
    except requests.exceptions.RequestException:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def search_professors(query):
    """Search for professors by name"""
    try:
        response = requests.get(f"{PLANETTERP_BASE_URL}/search", 
                               params={"query": query, "type": "professor"}, 
                               timeout=10)
        return response.json() if response.status_code == 200 else []
    except requests.exceptions.RequestException:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_course_details(course_id):
    """Get detailed information about a course"""
    try:
        response = requests.get(f"{PLANETTERP_BASE_URL}/course", 
                               params={"name": course_id}, 
                               timeout=10)
        return response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_professor_details(professor_name):
    """Get detailed information about a professor"""
    try:
        response = requests.get(f"{PLANETTERP_BASE_URL}/professor", 
                               params={"name": professor_name}, 
                               timeout=10)
        return response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_reviews(type_param, name):
    """Get reviews for a course or professor"""
    try:
        response = requests.get(f"{PLANETTERP_BASE_URL}/reviews", 
                               params={"type": type_param, "name": name}, 
                               timeout=10)
        return response.json() if response.status_code == 200 else []
    except requests.exceptions.RequestException:
        return []

def extract_entities(user_input):
    """Try to extract course IDs and professor names from input"""
    # Look for course patterns (e.g., CMSC131, ENGL101)
    import re
    course_pattern = r'\b[A-Z]{4}\s*\d{3}\b'
    courses = re.findall(course_pattern, user_input.upper())
    
    # For professors, we'll just use search endpoint
    return [c.replace(" ", "") for c in courses]

def format_data_for_llm(data):
    """Format data in a way that's useful for the LLM"""
    # Remove excessively large fields
    if isinstance(data, dict):
        if "reviews" in data and len(data["reviews"]) > 3:
            data["reviews"] = data["reviews"][:3]  # Only first 3 reviews
            data["reviews"].append({"note": f"{len(data['reviews']) - 3} more reviews available"})
    return data

def generate_response(user_input, data_context):
    """Generate response using Gemini API"""
    try:
        # Create system prompt for context
        system_prompt = """
        You are a helpful assistant for University of Maryland students using PlanetTerp data.
        Provide insights about courses and professors based on the data provided.
        Focus on:
        1. Grade distributions and average GPA
        2. Professor ratings and reviews
        3. Course difficulty and workload
        4. Recommendations based on past students' experiences
        
        Always cite the source as PlanetTerp data. If data is missing or limited, acknowledge that.
        Use a friendly, informative tone suitable for students.
        """
        
        # Format the data context for the model
        formatted_context = format_data_for_llm(data_context)
        
        # Create the user message with context
        user_message = f"""
        PlanetTerp Data: {json.dumps(formatted_context, indent=2)}
        
        Student Question: {user_input}
        
        Please provide a helpful response based on this PlanetTerp data.
        """
        
        # Add system instructions to chat
        response = st.session_state.chat.send_message(
            system_prompt + "\n\n" + user_message
        )
        
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def create_grade_chart(grades):
    """Create a visualization for grade distribution"""
    if not grades:
        return None
    
    grade_order = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F', 'W', 'Other']
    grade_data = []
    
    for grade in grade_order:
        if grade in grades and grades[grade] > 0:
            grade_data.append({"Grade": grade, "Percentage": grades[grade]})
        elif grade == 'Other':
            other_sum = sum(grades.get(k, 0) for k in grades if k not in grade_order)
            if other_sum > 0:
                grade_data.append({"Grade": "Other", "Percentage": other_sum})
    
    if not grade_data:
        return None
        
    df = pd.DataFrame(grade_data)
    fig = px.bar(df, x="Grade", y="Percentage", 
                 title="Grade Distribution",
                 color="Grade",
                 color_discrete_sequence=px.colors.sequential.Blues_r)
    return fig

# Streamlit UI
st.set_page_config(page_title="PlanetTerp Chatbot", page_icon="🐢", layout="wide")

# Header and intro
st.title("🐢 PlanetTerp Chatbot")
st.markdown("""
Ask questions about University of Maryland courses and professors! 
Examples:
- "Tell me about CMSC131"
- "Who is the best professor for MATH140?"
- "What's the grade distribution for PSYC100?"
""")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display charts if available
        if message["role"] == "assistant" and "chart" in message:
            st.plotly_chart(message["chart"])

# Chat input
if user_input := st.chat_input("Ask about UMD courses/professors..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.spinner("Analyzing PlanetTerp data..."):
        # Extract entities from user input
        possible_courses = extract_entities(user_input)
        
        # Initialize data context
        data_context = {
            "courses": [],
            "professors": [],
            "reviews": []
        }
        
        # Search for courses first
        course_results = search_courses(user_input)
        if course_results:
            for course in course_results[:2]:  # Limit to top 2 matches
                course_id = course.get("course_id") or course.get("name")
                if course_id:
                    course_details = get_course_details(course_id)
                    if course_details:
                        data_context["courses"].append(course_details)
                        course_reviews = get_reviews("course", course_id)
                        if course_reviews:
                            data_context["reviews"].extend(course_reviews[:5])  # Limit reviews
        
        # Then check for specific courses mentioned
        for course_id in possible_courses:
            if not any(c.get("course_id") == course_id for c in data_context["courses"]):
                course_details = get_course_details(course_id)
                if course_details:
                    data_context["courses"].append(course_details)
                    course_reviews = get_reviews("course", course_id)
                    if course_reviews:
                        data_context["reviews"].extend(course_reviews[:5])  # Limit reviews
        
        # Search for professors
        professor_results = search_professors(user_input)
        if professor_results:
            for prof in professor_results[:2]:  # Limit to top 2 matches
                prof_name = prof.get("name")
                if prof_name:
                    prof_details = get_professor_details(prof_name)
                    if prof_details:
                        data_context["professors"].append(prof_details)
                        prof_reviews = get_reviews("professor", prof_name)
                        if prof_reviews:
                            data_context["reviews"].extend(prof_reviews[:5])  # Limit reviews
        
        # Generate response with Gemini
        bot_response = generate_response(user_input, data_context)
        
        # Create visualization if we have grade data
        chart = None
        for course in data_context.get("courses", []):
            if course and "grades" in course and course["grades"]:
                chart = create_grade_chart(course["grades"])
                break
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(bot_response)
            if chart:
                st.plotly_chart(chart)
        
        # Add to message history
        message_with_chart = {"role": "assistant", "content": bot_response}
        if chart:
            message_with_chart["chart"] = chart
        st.session_state.messages.append(message_with_chart)

# Sidebar with additional info
with st.sidebar:
    st.header("About PlanetTerp Chatbot")
    st.markdown("""
    This chatbot uses data from [PlanetTerp](https://planetterp.com) to help UMD students 
    learn about courses and professors. It leverages the Google Gemini AI API to provide 
    helpful insights based on grade distributions and student reviews.
    """)
    
    st.header("Raw Data Preview")
    if len(st.session_state.messages) > 1:
        with st.expander("View Latest API Data"):
            if 'data_context' in locals():
                st.json(data_context)
            else:
                st.write("No data available")