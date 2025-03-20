import requests
import numpy as np
import re
import os
import json
import datetime
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from dotenv import load_dotenv

# Basic setup
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
PLANETTERP_BASE_URL = "https://api.planetterp.com/v1"

# Core API functions
def get_courses():
    response = requests.get(f"{PLANETTERP_BASE_URL}/courses")
    if response.status_code == 200:
        return response.json()
    return []

def get_course(course_id):
    response = requests.get(f"{PLANETTERP_BASE_URL}/course", params={"name": course_id})
    if response.status_code == 200:
        return response.json()
    return None

def get_professor(name):
    response = requests.get(f"{PLANETTERP_BASE_URL}/professor", 
                           params={"name": name, "reviews": "true"})
    if response.status_code == 200:
        professor_data = response.json()
        # Sort reviews by recency
        return sort_professor_reviews(professor_data)
    return None

def sort_professor_reviews(professor_data):
    if professor_data and 'reviews' in professor_data and professor_data['reviews']:
        # Sort reviews by date in descending order
        professor_data['reviews'].sort(key=lambda x: x.get('created', ''), reverse=True)
    return professor_data

def filter_recent_grades(grades, years=4):
    # Get current year
    current_year = datetime.datetime.now().year
    
    # Filter grades from the last 'years' years
    recent_grades = [g for g in grades if g.get('semester') and 
                    int(g.get('semester', '000000')[:4]) >= (current_year - years)]
    
    # Sort by semester in descending order (most recent first)
    recent_grades.sort(key=lambda x: x.get('semester', '000000'), reverse=True)
    
    return recent_grades

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

# Model and semantic search functions
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def initialize_index(model, courses):
    if not courses:
        return None, []
        
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
    
    return index, ids

def search(query, index, course_ids, model, k=5):
    if index is None:
        return []
    
    query_embedding = model.encode([query])
    _, indices = index.search(
        np.array(query_embedding).astype('float32'), k
    )
    
    results = []
    for idx in indices[0]:
        if idx < len(course_ids):
            course_id = course_ids[idx]
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
def generate_response(query, data, chat_model):
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
    
    response = chat_model.send_message(system + prompt)
    return response.text

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

def get_greeting():
    hour = datetime.datetime.now().hour
    if hour > 5 and hour < 11:
        return "Good morning"
    elif hour >= 11 and hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"