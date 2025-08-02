from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from planetterp_core import (
    get_courses, get_course, get_professor, get_course_grades, extract_course_ids,
    load_model, initialize_index, search, generate_response
)

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    context: Dict[str, Any] = {}

# Cache model and index in memory
model = None
index = None
course_ids = []
courses_data = None

def ensure_index():
    global model, index, course_ids, courses_data
    if model is None:
        model = load_model()
    if courses_data is None:
        courses_data = get_courses()
    if index is None or not course_ids:
        index, course_ids = initialize_index(model, courses_data)

@app.get("/courses")
def courses():
    return get_courses()

@app.get("/course/{course_id}")
def course(course_id: str):
    return get_course(course_id)

@app.get("/professor/{name}")
def professor(name: str):
    return get_professor(name)

@app.get("/grades/{course_id}")
def grades(course_id: str):
    return get_course_grades(course_id)

@app.post("/chat")
def chat(req: ChatRequest):
    ensure_index()
    query = req.message
    # Extract course IDs
    direct_course_ids = extract_course_ids(query)
    semantic_course_ids = [] if direct_course_ids else search(
        query, index, course_ids, model
    )
    all_course_ids = direct_course_ids + semantic_course_ids
    data = {"courses": [], "professors": [], "grades": []}
    for course_id in all_course_ids[:3]:
        course = get_course(course_id)
        if course:
            data["courses"].append(course)
            grades = get_course_grades(course_id)
            if grades:
                data["grades"].extend(grades[:5])
            for prof_name in course.get("professors", [])[:5]:
                prof = get_professor(prof_name)
                if prof:
                    data["professors"].append(prof)
    # Generate response
    class DummyChat:
        def send_message(self, prompt):
            # For Gemini, you would use the real chat model. For now, just echo.
            return type("Resp", (), {"text": "(Backend not fully implemented: " + prompt[:100] + ")"})()
    # Use the real chat model if available, else dummy
    from planetterp_core import genai
    try:
        chat_model = genai.GenerativeModel(
            'gemini-2.0-flash',
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024,
            },
        ).start_chat(history=[])
    except Exception:
        chat_model = DummyChat()
    response = generate_response(query, data, chat_model)
    return {"response": response} 