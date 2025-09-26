# PlanetTerpChatBot

## 🚀 Try It Live

👉 [**Launch PlanetTerpChatBot**](https://planetterp-chat-bot.streamlit.app/)

PlanetTerpChatBot is a Streamlit-based chatbot built to enhance how University of Maryland (UMD) students interact with academic resources available on PlanetTerp. Instead of using traditional search tools, users can engage in a conversational interface to explore course reviews, professor ratings, grade distributions, and more.

---

## 🚀 Project Overview

The **PlanetTerp Chatbot** is an AI-powered virtual assistant designed to improve the user experience of the [PlanetTerp](https://planetterp.com/) website — a platform where UMD students access reviews and ratings of courses, professors, and historical grade distributions.

Unlike traditional search tools, this chatbot provides a fully **interactive and immersive experience**, allowing users to:

- Ask academic questions naturally
- Get real-time results from PlanetTerp data
- Retrieve course reviews, professor ratings, and grade history
- Discover fun facts about UMD
- View a sidebar of essential UMD-related resources
- Maintain chat history for context-aware conversations

With its AI capabilities and semantic search, the chatbot helps UMD students make better-informed decisions about their courses and professors.

---

## 🛠️ Technical Details

- **Frontend**: Built with [Streamlit](https://streamlit.io) for a responsive, easy-to-use web UI  
- **Backend**: Python-based logic for handling query processing and response generation  
- **AI Integration**: Utilizes the **Google Gemini API** for generating precise, context-aware responses  
- **Semantic Search**: Delivers more relevant results by understanding the intent behind user queries  
- **Caching**: Implements caching strategies to reduce latency and improve responsiveness  
- **State Management**: Supports chat history to preserve conversation context across interactions

---

## 📋 Table of Contents

- [Features](#features)  
- [Architecture / Components](#architecture--components)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Running Locally](#running-locally)  
- [Usage](#usage)  
- [Configuration](#configuration)  
- [Project Structure](#project-structure)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact / Acknowledgments](#contact--acknowledgments)

---

## ✨ Features

- Conversational search for UMD course and professor data  
- Real-time retrieval of PlanetTerp content  
- Chat history for context-aware conversations  
- Sidebar with helpful UMD resources  
- Fun facts and trivia about UMD  
- Fast response times with intelligent caching  
- Semantic search for accurate and relevant results  

---

## 🧱 Architecture / Components

- `app.py`: Streamlit UI & interaction logic  
- `planetterp_core.py`: Chatbot core logic and query handling  
- `requirements.txt`: List of Python dependencies  

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.8 or higher  
- pip (Python package manager)

### Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/narainsriram2020/PlanetTerpChatBot.git
   cd PlanetTerpChatBot
   ```

2. (Optional) Create & activate a virtual environment  
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS / Linux
   venv\Scripts\activate       # Windows
   ```

3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

### Running Locally

```bash
streamlit run app.py
```

This will open the chatbot in your default browser at `http://localhost:8501`.

---

## 💬 Usage

1. Launch the chatbot via the Streamlit app  
2. Enter a query like:
   - *"What do students say about CMSC131?"*
   - *"How is Professor X for MATH140?"*
3. Get AI-generated responses backed by PlanetTerp data  
4. Use the sidebar to access quick links and resources  
5. Continue the conversation — context is preserved!

---

## 🔧 Configuration

Currently, the app runs without external configuration, but you can add:

- `.env` file for API keys or secrets (e.g., Gemini API key)
- `config.py` for modular settings
- Environment variable handling via `os.environ`

---

## 📁 Project Structure

```
PlanetTerpChatBot/
├── .devcontainer/
├── .idea/
├── app.py
├── planetterp_core.py
├── requirements.txt
├── README.md
├── .gitignore
└── __pycache__/
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repo  
2. Create a branch: `git checkout -b feature/YourFeature`  
3. Make changes and commit: `git commit -m "Add feature X"`  
4. Push your branch: `git push origin feature/YourFeature`  
5. Submit a Pull Request

---

## 📄 License

*This project currently does not specify a license.*  
To encourage collaboration and usage, consider adding an open-source license (e.g., MIT, Apache 2.0).

---

## 👤 Contact / Acknowledgments

- **Author**: [@narainsriram2020](https://github.com/narainsriram2020)  
- **Live App**: [https://planetterp-chat-bot.streamlit.app](https://planetterp-chat-bot.streamlit.app)  
- **Special Thanks**: PlanetTerp for their public data and UMD students for feedback and testing

---

