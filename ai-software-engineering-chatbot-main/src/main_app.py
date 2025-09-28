from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
from config import get_model

variable = get_model()

DEFAULT_CONFIG = {
    "speech_to_text": {
        "enabled": True,
        "language": "en-US",
    },
}

# Keywords related to software engineering
SOFTWARE_ENGINEERING_KEYWORDS = [
    "aiml", "artificial intelligence", "machine learning", "deep learning", "ai","IT",
    "natural language processing", "nlp", "data science", "unix", "linux", 
    "shell scripting", "bash", "python", "java", "c", "c++", "programming languages", 
    "object-oriented programming", "functional programming", "software development", 
    "software engineering", "algorithms", "data structures", "version control", 
    "git", "github", "software testing", "test-driven development", "tdd", "continuous integration", 
    "continuous deployment", "agile methodology", "scrum", "devops", "docker", 
    "kubernetes", "cloud computing", "aws", "azure", "google cloud platform", 
    "web development", "frontend development", "backend development", "html", 
    "css", "javascript", "node.js", "react.js", "angular", "vue.js", 
    "mobile development", "android", "ios", "swift", "kotlin", 
    "database management", "sql", "nosql", "mongodb", "mysql", 
    "postgresql", "restful apis", "graphql", "microservices", 
    "software architecture", "design patterns", "cybersecurity", 
    "encryption", "networking", "tcp/ip", "udp", "http", "https", 
    "socket programming", "distributed systems", "cloud-native", 
    "serverless computing", "functional programming", 
    "web scraping", "data visualization", "matplotlib", 
    "seaborn", "plotly", "pandas", "numpy", "scipy", 
    "opencv", "tensorflow", "keras", "pytorch", 
    "scikit-learn", "java virtual machine", 
    "object-oriented design", "design patterns", 
    "c programming", "c++ programming"    
]

# Keywords related to other branches of engineering
OTHER_ENGINEERING_KEYWORDS = [
    "mechanical engineering", "electrical engineering", "civil engineering",  # Add more branches as needed
    # Include keywords specific to each engineering branch
]

@cl.on_chat_start
async def on_chat_start():
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=variable, convert_system_message_to_human=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You have an extreme knowledge in the field of software engineering. If your query is about software engineering, please include related keywords. Otherwise, specify the branch of engineering you're interested in.""",
            ),
            ("human", "{question}"),
        ]
    )
    user_config = cl.user_session.get("config", {})
    merged_config = {**DEFAULT_CONFIG, **user_config}
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")
    
    # Check if the message contains any engineering-related keywords
    contains_software_engineering_keyword = any(keyword in message.content.lower() for keyword in SOFTWARE_ENGINEERING_KEYWORDS)
    contains_other_engineering_keyword = any(keyword in message.content.lower() for keyword in OTHER_ENGINEERING_KEYWORDS)

    if contains_software_engineering_keyword or contains_other_engineering_keyword:
        async for chunk in runnable.astream(
            {"question": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)

        await msg.send()
    else:
        await message.reply("Please ask relevant questions related to engineering.")
