# Software Engineering Concept AI Chatbot

## 🤖 Project Overview
An intelligent chatbot designed to provide comprehensive insights and answers to software engineering concepts using Chainlit and advanced API integration.

## ✨ Features
- Interactive conversational interface
- Specialized in software engineering topics
- Leverages advanced API for intelligent responses
- Easy to extend and customize

## 🛠 Tech Stack
- Python
- Chainlit
- API Integration
- (Google API KEY)

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/Harrie07/ai-software-engineering-chatbot.git

# Navigate to project directory
cd ai-software-engineering-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API credentials
```

## 📦 Project Structure
- `src/config.py`: Configuration management
- `src/main_app.py`: Core Chainlit application
- `tests/`: Unit tests for the application

## 🔧 Configuration
Edit `src/config.py` to customize:
- API endpoints
- Response parameters
- Logging settings

## 🚀 Running the Chatbot
```bash
# Activate virtual environment
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Run the Chainlit app
chainlit run src/main_app.py
```

## 💡 Example Usage
```python
# Interact with the chatbot through the Chainlit interface
# Ask questions about software engineering concepts
# Get detailed, context-aware responses
```

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request


## 📞 Contact
Harshal Sakpal - [harshalsakpal21@gmail.com](mailto:harshalsakpal21@gmail.com)

Project Link: [https://github.com/Harrie07/ai-software-engineering-chatbot](https://github.com/Harrie07/ai-software-engineering-chatbot)
