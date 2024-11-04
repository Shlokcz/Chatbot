import streamlit as st
import requests
import json
import logging
from datetime import datetime, timezone
import uuid
import bcrypt
import streamlit_authenticator as stauth
import os
import certifi
from pymongo import MongoClient
from dotenv import load_dotenv
import time

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.ERROR) 

# Set the page configuration
st.set_page_config(
    page_title="📄 PDF Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for chat history and loaded history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'loaded_history' not in st.session_state:
    st.session_state['loaded_history'] = []
    st.session_state['show_saved_history'] = False  # Flag to track visibility
    

# MongoDB Connection using environment variable and certifi
MONGODB_URI = os.getenv("MONGODB_URI")  # Fetch the URI from environment variables
try:
    client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    db = client["chatbotDB"]
    users_collection = db["users"]  # The collection where users are stored
    # success_message = st.success("Connected to MongoDB Atlas!")
    # if(success_message):
    #     time.sleep(1)
    #     success_message.empty()
except Exception as e:
    st.error(f"Failed to connect to MongoDB: {e}")


# Function to fetch user credentials from MongoDB
def get_user(username):
    user = users_collection.find_one({"username": username})
    return user

# Function to hash a password
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Function to verify the password
def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

# Authentication Functionality
def authenticate_user(username, password):
    user = get_user(username)
    if user:
        hashed_password = user["hashed_password"]
        if verify_password(password, hashed_password):
            return True
    return False

#Pasting the send_message function
def send_message(message: str) -> dict:
    """
    Sends a chat message to the backend and retrieves the response.
    
    Parameters:
    - message (str): The user's input message.
    
    Returns:
    - dict: Contains 'message_id' and 'answer' from the chatbot.
    """
    payload = {"query": message}
    try:
        response = requests.post("http://localhost:8000/chat/", json=payload)
        if response.status_code == 200:
            data = response.json()
            return data  # Should contain 'message_id' and 'answer'
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return {"message_id": str(uuid.uuid4()), "answer": f"Error {response.status_code}: {error_detail}"}
    except requests.exceptions.ConnectionError:
        return {"message_id": str(uuid.uuid4()), "answer": "Failed to connect to the backend. Make sure the FastAPI server is running on http://localhost:8000."}
    except Exception as e:
        logging.error(f"Unexpected error during chat: {e}")
        return {"message_id": str(uuid.uuid4()), "answer": f"An unexpected error occurred: {e}"}
    
#Pasting the submit_feedback function
def submit_feedback(message_id: str, feedback: str):
    """
    Sends feedback to the backend for a specific message.
    
    Parameters:
    - message_id (str): The unique identifier for the message.
    - feedback (str): The feedback provided by the user ('Like' or 'Dislike').
    """
    feedback_data = {
        "message_id": message_id,
        "feedback": feedback
    }
    try:
        response = requests.post("http://localhost:8000/feedback/", json=feedback_data)
        if response.status_code == 200:
            st.success(f"Feedback '{feedback}' recorded.")
            # Update the local chat history with feedback
            for chat in st.session_state['chat_history']:
                if chat['message_id'] == message_id:
                    chat['feedback'] = feedback
                    break
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.error(f"Failed to record feedback: {error_detail}")
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the backend. Make sure the FastAPI server is running on http://localhost:8000.")
    except Exception as e:
        logging.error(f"Unexpected error during feedback submission: {e}")
        st.error(f"An unexpected error occurred: {e}")    

def chatbot_interface():
    # Title of the App
            st.title("📄 PDF Chatbot")
            st.markdown("Upload a PDF and start chatting with your documents!")

            # Initialize session state for chat history and loaded history
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []

            if 'loaded_history' not in st.session_state:
                st.session_state['loaded_history'] = []
                st.session_state['show_saved_history'] = False  # Flag to track visibility

            # ============================
            # PDF Upload Section
            # ============================
            st.header("📥 Upload PDF")

            uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

            if uploaded_files:
                if st.button("Upload and Process PDF"):
                    with st.spinner("Uploading and processing..."):
                        files = []
                        for uploaded_file in uploaded_files:
                            files.append(('file', (uploaded_file.name, uploaded_file, 'application/pdf')))  # Use 'file' as the key
                        try:
                            response = requests.post("http://localhost:8000/upload-pdf/", files=files)
                            if response.status_code == 200:
                                data = response.json()
                                st.success(data['message'])
                                st.info(f"Vector IDs: {', '.join(data['vector_ids'])}")
                            else:
                                error_detail = response.json().get('detail', 'Unknown error')
                                st.error(f"Error {response.status_code}: {error_detail}")
                        except requests.exceptions.ConnectionError:
                            st.error("Failed to connect to the backend. Make sure the FastAPI server is running on http://localhost:8000.")
                        except Exception as e:
                            logging.error(f"Unexpected error during PDF upload: {e}")
                            st.error(f"An unexpected error occurred: {e}")

            # ============================
            # Chat Section
            # ============================
            st.header("💬 Chat with Your PDF")

            # Input box for user queries
            user_input = st.text_input("You:", "")

            # Send button functionality
            if st.button("Send") and user_input:
                with st.spinner("Validating and generating response..."):
                    # Send the message to the backend and get the response
                    response = send_message(user_input)
                    message_id = response.get('message_id', str(uuid.uuid4()))
                    bot_response = response.get('answer', "No response from the bot.")
                    # Generate a timestamp for the conversation entry
                    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    # Append the new conversation to the chat history
                    st.session_state['chat_history'].append({
                        "message_id": message_id,
                        "timestamp": timestamp,
                        "user": user_input,
                        "bot": bot_response,
                        "feedback": None
                    })
                    # Debugging: Print the last message added
                    logging.debug(f"Message added to history: {st.session_state['chat_history'][-1]}")


            # ============================
            # Latest Conversation Section
            # ============================
            st.header("🆕 Latest Conversation")

            # Display the most recent conversation if available
            if st.session_state['chat_history']:
                latest_chat = st.session_state['chat_history'][-1]  # Get the last entry
                # Safeguard against missing keys
                user_text = latest_chat.get('user', 'N/A')
                bot_text = latest_chat.get('bot', 'N/A')
                # intent_text = latest_chat.get('intent', 'N/A') 
                timestamp = latest_chat.get('timestamp', 'N/A')
                st.markdown(f"**[{timestamp}] You:** {user_text}")
                st.markdown(f"**Bot:** {bot_text}\n")
                # st.markdown(f"**Intent:** {user_input}\n") 
                # Add Like and Dislike buttons for the latest response 
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Like", key=f"like_{latest_chat['message_id']}"):
                        submit_feedback(message_id=latest_chat['message_id'], feedback="Like")
                with col2:
                    if st.button("👎 Dislike", key=f"dislike_{latest_chat['message_id']}"):
                        submit_feedback(message_id=latest_chat['message_id'], feedback="Dislike")
            else:
                st.info("No conversations yet. Start by sending a message!")


            # ============================
            # Conversation History Section
            # ============================
            st.header("📜 Conversation History")

            # Display all past conversations excluding the latest one
            if st.session_state['chat_history']:
                if len(st.session_state['chat_history']) > 1:
                    st.markdown("### All Past Conversations")
                    # Reverse the list to show the most recent past conversations first
                    for chat in reversed(st.session_state['chat_history'][:-1]):
                        # Safeguard against missing keys
                        user_text = chat.get('user', 'N/A')
                        bot_text = chat.get('bot', 'N/A')
                        timestamp = chat.get('timestamp', 'N/A')
                        st.markdown(f"**[{timestamp}] You:** {user_text}")
                        st.markdown(f"**Bot:** {bot_text}")
                        if chat.get('feedback'):
                            st.markdown(f"**Feedback:** {chat['feedback']}\n")
                        else:
                            # Add Like and Dislike buttons for each past response
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("👍 Like", key=f"like_{chat['message_id']}"):
                                    submit_feedback(message_id=chat['message_id'], feedback="Like")
                            with col2:
                                if st.button("👎 Dislike", key=f"dislike_{chat['message_id']}"):
                                    submit_feedback(message_id=chat['message_id'], feedback="Dislike")
                                    st.markdown("---")
                else:
                    st.info("Only one conversation exists.")
            else:
                st.info("No conversation history found.")    


            # ============================
            # Load Conversation History from Backend 
            # ============================
            st.header("📂 Load Saved Conversation History")

            # Toggle Button: Load or Hide
            if not st.session_state['show_saved_history']:
                load_button = st.button("Load Saved Conversation History")
                if load_button:
                    with st.spinner("Fetching conversation history..."):
                        try:
                            response = requests.get("http://localhost:8000/conversation-history/")
                            if response.status_code == 200:
                                history = response.json()
                                if history:
                                    # Update session state with loaded history
                                    st.session_state['loaded_history'] = history
                                    st.session_state['show_saved_history'] = True
                                    st.success("Conversation history loaded successfully!")
                                    # Prepend loaded history to chat_history with key mapping
                                    for entry in reversed(history):  # Reverse to maintain chronological order
                                        # Avoid duplicating the latest chat if it's already in chat_history
                                        if not any(chat['message_id'] == entry['message_id'] for chat in st.session_state['chat_history']):
                                            st.session_state['chat_history'].insert(0, {
                                            "message_id": entry['message_id'],
                                            "timestamp": entry['timestamp'],
                                            "user": entry['user_query'],          # Map 'user_query' to 'user'
                                            "bot": entry['bot_response'],         # Map 'bot_response' to 'bot'
                                            "feedback": entry['feedback']
                                        })
                                else:
                                    st.info("No conversation history found on the backend.")
                            else:
                                error_detail = response.json().get('detail', 'Unknown error')
                                st.error(f"Error {response.status_code}: {error_detail}")
                        except requests.exceptions.ConnectionError:
                            st.error("Failed to connect to the backend. Make sure the FastAPI server is running on http://localhost:8000.")
                        except Exception as e:
                            logging.error(f"Unexpected error while loading conversation history: {e}")
                            st.error(f"An unexpected error occurred: {e}")
            else:
                hide_button = st.button("Hide Saved Conversation History")
                if hide_button:
                    st.session_state['show_saved_history'] = False
                    st.success("Saved conversation history hidden.")

            # ============================
            # Display Loaded Conversation History
            # ============================
            if st.session_state['show_saved_history'] and st.session_state['loaded_history']:
                st.markdown("### All Past Conversations from Backend")
                for entry in st.session_state['loaded_history']:
                    timestamp = entry.get("timestamp", "N/A")
                    user_query = entry.get("user_query", "")
                    bot_response = entry.get("bot_response", "")
                    feedback = entry.get("feedback", None)
                    st.markdown(f"**[{timestamp}] You:** {user_query}")
                    st.markdown(f"**Bot:** {bot_response}")
                    if feedback:
                        st.markdown(f"**Feedback:** {feedback}\n")
                    else:
                        # Add Like and Dislike buttons for each loaded response
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("👍 Like", key=f"like_loaded_{entry.get('message_id', 'loaded')}"):
                                submit_feedback(message_id=entry.get('message_id', 'loaded'), feedback="Like")
                        with col2:
                            if st.button("👎 Dislike", key=f"dislike_loaded_{entry.get('message_id', 'loaded')}"):
                                submit_feedback(message_id=entry.get('message_id', 'loaded'), feedback="Dislike")
                    st.markdown("---")   


            # ============================
            # Sidebar for Additional Navigation
            # ============================
            st.sidebar.header("📄 PDF Chatbot")
            st.sidebar.markdown("Upload PDFs and chat with your documents!")  

            # ============================
            # Sidebar: Clear Chat History Button
            # ============================
            if st.sidebar.button("Clear Chat History"):
                st.session_state['chat_history'] = []
                st.sidebar.success("Chat history cleared.")            

            # ============================
            # Sidebar: Download Chat History Button
            # ============================
            if st.sidebar.button("Download Chat History"):
                if st.session_state['chat_history']:
                    # Convert chat history to JSON format for downloading
                    chat_json = json.dumps(st.session_state['chat_history'], indent=2)
                    # Provide a download button for the JSON file
                    st.sidebar.download_button(
                        label="Download Chat History",
                        data=chat_json,
                        file_name="chat_history.json",
                        mime="application/json"
                    )
                    st.sidebar.success("Chat history ready for download.")
                else:
                    st.sidebar.info("No chat history to download.")

# Streamlit Authentication System
def login():

    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    # If authenticated, display the chatbot interface
    if st.session_state['authenticated']:
        chatbot_interface()
    else:
        # Show login form if not authenticated
        st.title("Login Page")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
    
    # st.title("Login Page")

    # # Input for login
    # username = st.text_input("Username")
    # password = st.text_input("Password", type="password")

        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state['authenticated'] = True  # Set authenticated to True
                st.success("Login successful!")  # Show success message
                st.rerun()  # Rerun the app to show the chatbot interface
                # st.success(f"Hi {username}, thank you for logging in!")
                
                # chatbot_interface()  # Calling the chatbot functionality here

                # st.markdown(f"<h1 style='text-align: center;'>Thank You So Much for Logging In!</h1>", unsafe_allow_html=True)
                # st.markdown(f"<h2 style='text-align: center;'>Hope you have a great day!</h2>", unsafe_allow_html=True)
            else:
                st.error("Invalid username or password. Please try again or register.")

# Registration functionality for new users
def register():
    st.title("Register Page")

    name = st.text_input("Create Name")
    username = st.text_input("Create Username")
    email = st.text_input("Email")
    password = st.text_input("Create Password", type="password")
    password_confirm = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if password == password_confirm:
            # Check if user already exists
            if get_user(username):
                st.error("Username already exists. Please try a different one.")
            else:
                # Hash the password and store it in MongoDB
                hashed_password = hash_password(password)
                users_collection.insert_one({
                    "username": username,
                    "email": email,
                    "name": name,
                    "hashed_password": hashed_password
                })
                st.success("User registered successfully! Please login.")
        else:
            st.error("Passwords do not match. Please try again.")

# Main function to run the app
def main():
    st.sidebar.title("Navigation")
    options = ["Login", "Register"]
    choice = st.sidebar.radio("Select an option", options)

    if choice == "Login":
        login()
    elif choice == "Register":
        register()

if __name__ == '__main__':
    main()
