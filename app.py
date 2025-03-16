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

# Enhanced session state with conversation context
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat" not in st.session_state:
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    st.session_state.chat = model.start_chat(history=[])
# Add conversation context to remember courses and professors
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = {
        "recent_courses": [],
        "recent_professors": [],
        "recent_questions": []
    }

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

def extract_entities(user_input):
    """Try to extract course IDs and professor names from input"""
    # Look for course patterns (e.g., CMSC131, ENGL101)
    import re
    course_pattern = r'\b[A-Z]{4}\s*\d{3}\b'
    courses = re.findall(course_pattern, user_input.upper())
    
    # For professors, we'll just use search endpoint
    return [c.replace(" ", "") for c in courses]

def update_conversation_context(user_input, data_context):
    """Update the conversation context with new information"""
    # Update recent questions
    st.session_state.conversation_context["recent_questions"].append(user_input)
    if len(st.session_state.conversation_context["recent_questions"]) > 5:
        st.session_state.conversation_context["recent_questions"].pop(0)
    
    # Update recent courses
    for course in data_context.get("courses", []):
        course_id = course.get("course_id") or course.get("name")
        if course_id:
            # Remove if already in list (to move to end)
            st.session_state.conversation_context["recent_courses"] = [
                c for c in st.session_state.conversation_context["recent_courses"] 
                if c["id"] != course_id
            ]
            
            # Add to end of list
            st.session_state.conversation_context["recent_courses"].append({
                "id": course_id,
                "name": course.get("title", ""),
                "avg_gpa": course.get("average_gpa", None),
                "professors": course.get("professors", [])
            })
    
    # Limit to 5 most recent courses
    if len(st.session_state.conversation_context["recent_courses"]) > 5:
        st.session_state.conversation_context["recent_courses"] = st.session_state.conversation_context["recent_courses"][-5:]
    
    # Update recent professors
    for prof in data_context.get("professors", []):
        prof_name = prof.get("name")
        if prof_name:
            # Remove if already in list (to move to end)
            st.session_state.conversation_context["recent_professors"] = [
                p for p in st.session_state.conversation_context["recent_professors"] 
                if p["name"] != prof_name
            ]
            
            # Add to end of list
            st.session_state.conversation_context["recent_professors"].append({
                "name": prof_name,
                "avg_rating": prof.get("average_rating", None),
                "courses": prof.get("courses", [])
            })
    
    # Limit to 5 most recent professors
    if len(st.session_state.conversation_context["recent_professors"]) > 5:
        st.session_state.conversation_context["recent_professors"] = st.session_state.conversation_context["recent_professors"][-5:]

def get_previous_context():
    """Get previous conversation context for the LLM"""
    context = {
        "recent_courses": st.session_state.conversation_context["recent_courses"],
        "recent_professors": st.session_state.conversation_context["recent_professors"],
        "recent_questions": st.session_state.conversation_context["recent_questions"]
    }
    return context

def format_data_for_llm(data):
    """Format data in a way that's useful for the LLM"""
    # Make a deep copy to avoid modifying the original
    import copy
    formatted_data = copy.deepcopy(data)
    
    # Add conversation context
    formatted_data["conversation_context"] = get_previous_context()
    
    return formatted_data

def generate_response(user_input, data_context):
    """Generate response using Gemini API"""
    try:
        # Create system prompt for context
        system_prompt = """
        You are a helpful assistant for University of Maryland students using PlanetTerp data.
        Provide insights about courses and professors based on the data provided.
        
        Focus on:
        1. Grade distributions and average GPA
        2. Professor ratings (without mentioning specific reviews)
        3. Course difficulty and workload based on available data
        4. Recommendations based on the data provided
        
        When the user asks follow-up questions, remember the context from previous questions.
        If a user asks about "this course" or "this professor", refer to the most recent course or professor in the conversation context.
        If a user asks about professors without specifying a course, check if there's a recently discussed course and provide professor information for that course.
        
        Always cite the source as PlanetTerp data. If data is missing or limited, acknowledge that.
        Use a friendly, informative tone suitable for students.
        """
        
        # Format the data context for the model
        formatted_context = format_data_for_llm(data_context)
        
        # Create the user message with context
        user_message = f"""
        PlanetTerp Data: {json.dumps(formatted_context, indent=2)}
        
        Student Question: {user_input}
        
        Please provide a helpful response based on this PlanetTerp data and the conversation context.
        DO NOT include any specific student reviews or quotes.
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

def create_rating_chart(professor_data):
    """Create a visualization for professor average rating"""
    if not professor_data or "average_rating" not in professor_data:
        return None
        
    rating = professor_data["average_rating"]
    
    # Create a gauge chart for rating
    fig = px.bar(
        x=["Rating"],
        y=[rating],
        labels={"x": "", "y": "Rating"},
        title=f"Professor Rating: {rating}/5.0",
        range_y=[0, 5],
        color_discrete_sequence=["#4682b4"]
    )
    
    # Add a horizontal line for the maximum rating
    fig.add_shape(
        type="line",
        x0=-0.5, y0=5, x1=0.5, y1=5,
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Add annotation for the maximum rating
    fig.add_annotation(
        x=0, y=5,
        text="Max Rating",
        showarrow=False,
        yshift=10
    )
    
    fig.update_layout(height=300)
    return fig

# Function to resolve ambiguous references
def resolve_references(user_input):
    """Resolve references like 'this course' or 'that professor'"""
    # Make a copy of the original input
    resolved_input = user_input
    
    # Check for ambiguous references
    ambiguous_terms = {
        "this course": None,
        "that course": None,
        "the course": None,
        "this class": None,
        "that class": None,
        "the class": None,
        "this professor": None,
        "that professor": None,
        "the professor": None,
        "this prof": None,
        "that prof": None, 
        "the prof": None
    }
    
    # Check if we have recent context
    if st.session_state.conversation_context["recent_courses"]:
        most_recent_course = st.session_state.conversation_context["recent_courses"][-1]
        course_id = most_recent_course["id"]
        
        # Replace course references
        for term in ["this course", "that course", "the course", "this class", "that class", "the class"]:
            if term.lower() in user_input.lower():
                resolved_input = resolved_input.replace(term, course_id)
    
    if st.session_state.conversation_context["recent_professors"]:
        most_recent_prof = st.session_state.conversation_context["recent_professors"][-1]
        prof_name = most_recent_prof["name"]
        
        # Replace professor references
        for term in ["this professor", "that professor", "the professor", "this prof", "that prof", "the prof"]:
            if term.lower() in user_input.lower():
                resolved_input = resolved_input.replace(term, prof_name)
    
    return resolved_input

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
- "Show me information about Professor Dennis Nola"
""")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display charts if available
        if message["role"] == "assistant" and "charts" in message:
            for chart in message["charts"]:
                if chart:
                    st.plotly_chart(chart)

# Continuing from with st.sidebar

    # Help section
    with st.expander("Tips for using this chatbot"):
        st.markdown("""
        - Be specific with course codes (e.g., CMSC131 instead of just 'intro to programming')
        - Include professor names if you want information about specific instructors
        - Ask about grade distributions, professor ratings, or course difficulty
        - Try questions like "Who is the easiest professor for MATH140?"
        - Use "Compare professors for MATH140" to see multiple instructors
        - You can ask follow-up questions like "What are the professors for this course?"
        - The chatbot remembers recent courses and professors you've discussed
        """)

# Chat input
if user_input := st.chat_input("Ask about UMD courses/professors..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.spinner("Analyzing PlanetTerp data..."):
        # Resolve references to previously mentioned courses/professors
        resolved_input = resolve_references(user_input)
        
        # Extract entities from resolved input
        possible_courses = extract_entities(resolved_input)
        
        # Initialize data context
        data_context = {
            "courses": [],
            "professors": []
        }
        
        # Track objects for visualization
        grade_chart = None
        rating_charts = []
        
        # Check if we need to use recently discussed courses/professors
        if not possible_courses and "professor" in user_input.lower() and st.session_state.conversation_context["recent_courses"]:
            # If asking about professors but no course specified, use the most recent course
            most_recent_course = st.session_state.conversation_context["recent_courses"][-1]
            course_id = most_recent_course["id"]
            course_details = get_course_details(course_id)
            if course_details:
                data_context["courses"].append(course_details)
                
                # Create grade chart if available
                if not grade_chart and course_details.get("grades"):
                    grade_chart = create_grade_chart(course_details["grades"])
        
        # Search for courses first
        course_results = search_courses(resolved_input)
        if course_results:
            for course in course_results[:2]:  # Limit to top 2 matches
                course_id = course.get("course_id") or course.get("name")
                if course_id:
                    course_details = get_course_details(course_id)
                    if course_details:
                        data_context["courses"].append(course_details)
                        
                        # Create grade chart for the first course with grade data
                        if not grade_chart and course_details.get("grades"):
                            grade_chart = create_grade_chart(course_details["grades"])
        
        # Then check for specific courses mentioned
        for course_id in possible_courses:
            if not any(c.get("course_id") == course_id or c.get("name") == course_id for c in data_context["courses"]):
                course_details = get_course_details(course_id)
                if course_details:
                    data_context["courses"].append(course_details)
                    
                    # Create grade chart for the first course with grade data
                    if not grade_chart and course_details.get("grades"):
                        grade_chart = create_grade_chart(course_details["grades"])
        
        # Search for professors
        professor_results = search_professors(resolved_input)
        if professor_results:
            for prof in professor_results[:2]:  # Limit to top 2 matches
                prof_name = prof.get("name")
                if prof_name:
                    prof_details = get_professor_details(prof_name)
                    if prof_details:
                        data_context["professors"].append(prof_details)
                        
                        # Create rating chart for professors with ratings
                        if prof_details.get("average_rating"):
                            rating_chart = create_rating_chart(prof_details)
                            if rating_chart:
                                rating_charts.append(rating_chart)
        
        # Also check professors mentioned in courses
        for course in data_context["courses"]:
            for prof_name in course.get("professors", [])[:3]:  # Limit to first 3 professors
                # Check if we already have this professor
                if not any(p.get("name") == prof_name for p in data_context["professors"]):
                    prof_details = get_professor_details(prof_name)
                    if prof_details:
                        data_context["professors"].append(prof_details)
                        
                        # Create rating chart if we have ratings and don't have too many charts already
                        if prof_details.get("average_rating") and len(rating_charts) < 2:
                            rating_chart = create_rating_chart(prof_details)
                            if rating_chart:
                                rating_charts.append(rating_chart)
        
        # Update conversation context with new data
        update_conversation_context(user_input, data_context)
        
        # Generate response with Gemini
        bot_response = generate_response(user_input, data_context)
        
        # Create charts list for display
        charts = []
        if grade_chart:
            charts.append(grade_chart)
        charts.extend(rating_charts[:2])  # Limit to 2 rating charts
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(bot_response)
            for chart in charts:
                if chart:
                    st.plotly_chart(chart)
        
        # Add to message history
        message_with_charts = {"role": "assistant", "content": bot_response}
        if charts:
            message_with_charts["charts"] = charts
        st.session_state.messages.append(message_with_charts)

# Add a debug expander (can be removed in production)
with st.sidebar:
    with st.expander("Debug Information", expanded=False):
        st.write("Session State:", st.session_state.conversation_context)