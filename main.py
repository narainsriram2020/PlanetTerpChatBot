from fastapi import FastAPI, Query
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Get API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PLANETTERP_API_KEY = os.getenv("PLANETTERP_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# PlanetTerp API base URL
PLANETTERP_API_BASE = "https://api.planetterp.com/v1"


# Fetch data from PlanetTerp API
def get_planetterp_data(endpoint: str, params: dict):
    response = requests.get(f"{PLANETTERP_API_BASE}/{endpoint}", params=params,
                            headers={"Authorization": f"Bearer {PLANETTERP_API_KEY}"})
    return response.json() if response.status_code == 200 else None


# Process user query
def process_query(query: str):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(query)
    return response.text if response else "I couldn't process that request."


# API Route for chatbot
@app.get("/chat")
def chat(query: str = Query(..., title="User query")):
    # Check if the query is about a professor, course, or grades
    if "professor" in query.lower():
        professor_name = query.split("professor")[-1].strip()
        data = get_planetterp_data("professors", {"name": professor_name})
        if data:
            return {"response": f"Here's what I found about {professor_name}: {data}"}

    elif "course" in query.lower():
        course_name = query.split("course")[-1].strip()
        data = get_planetterp_data("courses", {"name": course_name})
        if data:
            return {"response": f"Course info for {course_name}: {data}"}

    elif "grades" in query.lower():
        course_name = query.split("grades")[-1].strip()
        data = get_planetterp_data("grades", {"course": course_name})
        if data:
            return {"response": f"Grade distribution for {course_name}: {data}"}

    # Use Gemini for general queries
    return {"response": process_query(query)}


# Run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
