# main.py

import os
import uuid
import certifi
from typing import List, Optional
from datetime import datetime, timezone

import openai
import pinecone
import pdfplumber
import re
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer
import logging
from dotenv import load_dotenv  # For loading environment variables
import motor.motor_asyncio  # Asynchronous MongoDB driver
import warnings  # To handle warnings

# ============================
# Load Environment Variables
# ============================
load_dotenv()

# ============================
# Configure Logging
# ============================
logging.basicConfig(
    filename='chatbot.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Set to INFO to capture essential logs
)

# ============================
# Suppress Specific FutureWarnings
# ============================
# Suppress specific FutureWarning about clean_up_tokenization_spaces
warnings.filterwarnings(
    "ignore",
    message=r".*clean_up_tokenization_spaces.*",
    category=FutureWarning
)

# ============================
# Configuration and Constants
# ============================
# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logging.critical("OPENAI_API_KEY must be set as an environment variable.")
    raise ValueError("OPENAI_API_KEY must be set as an environment variable.")
openai.api_key = OPENAI_API_KEY  # Set OpenAI API key

# Pinecone Configuration
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')          # Fetch from environment variable
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')  # Fetch from environment variable

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    logging.critical("PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set as environment variables.")
    raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set as environment variables.")

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI')
if not MONGODB_URI:
    logging.critical("MONGODB_URI must be set as an environment variable.")
    raise ValueError("MONGODB_URI must be set as an environment variable.")

DATABASE_NAME = "chatbotDB"
CONVERSATION_COLLECTION = "conversation_history"
# FEEDBACK_COLLECTION is no longer needed
# FEEDBACK_COLLECTION = "feedback_history"

# Pinecone Index Configuration
INDEX_NAME = 'pdf-chatbot-index'
EMBEDDING_DIMENSION = 384  # Based on the sentence transformer model used

# ============================
# Initialize MongoDB
# ============================
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI, tlsCAFile=certifi.where())
db = client[DATABASE_NAME]
conversation_collection = db[CONVERSATION_COLLECTION]
# feedback_collection = db[FEEDBACK_COLLECTION]  # Not needed
logging.info("Connected to MongoDB Atlas.")

# ============================
# Initialize Pinecone
# ============================
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists; if not, create it
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',       # or your preferred cloud provider
                region='us-east-1'  # your desired region
            )
        )
        logging.info(f"Created Pinecone index '{INDEX_NAME}'.")
    except Exception as e:
        logging.critical(f"Failed to create Pinecone index '{INDEX_NAME}': {e}")
        raise e

# Connect to the index
index = pc.Index(INDEX_NAME)
logging.info(f"Connected to Pinecone index '{INDEX_NAME}'.")

# ============================
# Initialize Models
# ============================

# Initialize the Sentence Transformer model for embeddings without passing tokenizer
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight and efficient
embedding_model = SentenceTransformer(embedding_model_name)
logging.info(f"Loaded Sentence Transformer model '{embedding_model_name}'.")

# Verify embedding dimensions
sample_embedding = embedding_model.encode(["Sample text"], show_progress_bar=False)
logging.info(f"Embedding dimension: {len(sample_embedding[0])}")  # Should print 384

# ============================
# Define Pydantic Models
# ============================

class QueryRequest(BaseModel):
    query: str = Field(..., example="How can I raise my voice against harassment?")
    
    @field_validator('query')
    def query_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty.')
        return v

class PDFUploadResponse(BaseModel):
    message: str
    vector_ids: List[str] 

class FeedbackRequest(BaseModel):
    message_id: str = Field(..., example="123e4567-e89b-12d3-a456-426614174000")
    feedback: str = Field(..., example="Like")
    
    @field_validator('feedback')
    def feedback_must_be_valid(cls, v):
        if v not in {"Like", "Dislike"}:
            raise ValueError("Feedback must be either 'Like' or 'Dislike'.")
        return v

class ChatResponse(BaseModel):
    message_id: str
    answer: str

class ConversationLog(BaseModel):
    message_id: str = Field(..., example="123e4567-e89b-12d3-a456-426614174000")
    user_query: str
    intent: str
    bot_response: str
    feedback: Optional[str] = Field(None, example="Like")  # Optional field
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @field_validator('feedback')
    def validate_feedback(cls, v):
        if v not in {None, "Like", "Dislike"}:
            raise ValueError("Feedback must be either 'Like' or 'Dislike'.")
        return v

# ============================
# Utility Functions
# ============================

def clean_text(text: str) -> str:
    """
    Remove non-printable and special characters from text.
    """
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E]+', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract and clean text from a PDF file using pdfplumber.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    # Clean the extracted text
                    page_text = clean_text(page_text)
                    text += page_text + ' '
        return text.strip()
    except Exception as e:
        logging.error(f"Failed to extract and clean text from PDF '{pdf_path}': {e}")
        raise RuntimeError(f"Failed to extract and clean text from PDF: {e}")

def split_text(text: str, max_length: int = 500) -> List[str]:
    """
    Split text into chunks of approximately max_length characters.
    """
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < max_length:
            current_chunk += sentence + ' '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ' '
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using the Sentence Transformer model.
    """
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()

def validate_query(query: str) -> str:
    """
    Validates the user query using a simple check (you can customize this as needed).
    
    Parameters:
    - query (str): The user's input message.
    
    Returns:
    - str: 'Valid' if the query is appropriate, 'Invalid' otherwise.
    """
    # For simplicity, we can consider all queries as valid; you can add checks as necessary
    return "Valid" if query.strip() else "Invalid"

async def extract_intent(query: str) -> str:
    """
    Extracts the intent from the user query using OpenAI GPT-3.5-Turbo.
    
    Parameters:
    - query (str): The user's input message.
    
    Returns:
    - str: The identified intent as a simple string.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Please provide the intent for the following query: '{query}'."}
            ],
            max_tokens=50,
            temperature=0.0  # Lower temperature for more deterministic responses
        )
        intent = response.choices[0].message['content'].strip()
        logging.info(f"Extracted intent: {intent} for query: '{query}'")
        # print(response)
        return intent
    except openai.OpenAIError as e:
        logging.error(f"Error during intent extraction: {e}")
        return "Unknown"

async def log_conversation(message_id: str, user_query: str, bot_response: str, intent: str):
    """
    Log the conversation to MongoDB with intent and message_id.
    """
    conversation = ConversationLog(
        message_id=message_id,
        user_query=user_query,
        intent=intent,
        bot_response=bot_response
    )
    try:
        result = await conversation_collection.insert_one(conversation.model_dump())
        logging.info(f"Logged conversation successfully to MongoDB with _id: {result.inserted_id}.")
    except Exception as e:
        logging.error(f"Failed to log conversation to MongoDB: {e}")

# Removed log_feedback function as feedback is now part of ConversationLog

# ============================
# Initialize FastAPI
# ============================
app = FastAPI(title="PDF Chatbot API", version="1.0")

# ============================
# API Endpoints
# ============================

@app.post("/upload-pdf/", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF, extract its text, generate embeddings, and store them in Pinecone.
    """
    if file.content_type != 'application/pdf':
        logging.warning("Attempted to upload a non-PDF file.")
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are accepted.")

    # Save the uploaded PDF to a temporary file
    temp_filename = f"temp_{uuid.uuid4()}.pdf"
    try:
        with open(temp_filename, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        logging.info(f"Saved uploaded PDF as {temp_filename}.")
    except Exception as e:
        logging.error(f"Failed to save uploaded PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded PDF: {e}")

    # Extract text from the PDF
    try:
        text = extract_text_from_pdf(temp_filename)
        logging.info(f"Extracted text length: {len(text)} characters.")
    except RuntimeError as e:
        os.remove(temp_filename)  # Clean up
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

    os.remove(temp_filename)  # Clean up

    if not text.strip():
        logging.warning("No text found in the uploaded PDF.")
        raise HTTPException(status_code=400, detail="No text found in the PDF.")

    # Split text into chunks
    chunks = split_text(text)
    logging.info(f"Number of chunks created: {len(chunks)}.")

    # Generate embeddings for each chunk
    embeddings = get_embeddings(chunks)
    logging.info("Generated embeddings for chunks.")

    # Prepare vectors for Pinecone
    vectors = []
    vector_ids = []
    for chunk, embedding in zip(chunks, embeddings):
        vector_id = str(uuid.uuid4())
        vectors.append({
            'id': vector_id,
            'values': embedding,
            'metadata': {'text': chunk}
        })
        vector_ids.append(vector_id)

    # Upsert vectors into Pinecone
    try:
        index.upsert(vectors=vectors)
        logging.info(f"Uploaded PDF vectors: {vector_ids}")
    except Exception as e:
        logging.error(f"Failed to upsert vectors to Pinecone: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upsert vectors to Pinecone: {e}")

    return PDFUploadResponse(message="PDF processed and data stored successfully.", vector_ids=vector_ids)

@app.post("/chat/", response_model=ChatResponse)
async def chat(request: QueryRequest):
    """
    Endpoint to handle chat queries. It retrieves relevant information from Pinecone,
    performs intent extraction, and generates a response using OpenAI's ChatCompletion API.
    """
    query = request.query.strip()
    if not query:
        logging.warning("Received empty query.")
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Validate the query
    is_valid = validate_query(query)
    if is_valid != "Valid":
        logging.warning(f"Invalid query received: {is_valid}")
        raise HTTPException(status_code=400, detail=f"Invalid query: {is_valid}") 

    # Extract intent using GPT
    intent = await extract_intent(query)
    logging.info(f"Extracted intent: '{intent}'")

    # Generate embedding for the query
    query_embedding = get_embeddings([query])[0]
    logging.info(f"Generated embedding for query: '{query}'")

    # Query Pinecone for similar vectors
    try:
        response = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        logging.info(f"Pinecone query response: {response}")
    except Exception as e:
        logging.error(f"Failed to query Pinecone: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query Pinecone: {e}")

    if not response['matches']:
        logging.warning("No relevant documents found.")
        generated_text = "Sorry, I couldn't find relevant information."
        message_id = str(uuid.uuid4())
        await log_conversation(message_id=message_id, user_query=query, bot_response=generated_text, intent=intent)
        return ChatResponse(message_id=message_id, answer=generated_text)

    # Concatenate the text of the matched documents for context
    retrieved_texts = [match['metadata'].get('text', '') for match in response['matches']]
    context = "\n".join(retrieved_texts)
    logging.info(f"Retrieved context for GPT: '{context[:100]}...'")  # Log the first 100 characters

    # Prepare the prompt for OpenAI with intent
    messages = [
        {"role": "system", "content": "You are a knowledgeable assistant. Using the information provided in the context, answer the question accurately and concisely."},
        {"role": "user", "content": f"Context:\n{context}\n\nIntent: {intent}\n\nQuestion: {query}"}
    ]

    # Call OpenAI's API to generate a response
    try:
        response_openai = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use a supported ChatCompletion model
            messages=messages,
            max_tokens=150,  # Adjust based on desired response length
            temperature=0.7,
            top_p=0.9,
            n=1,
            stop=["\n"]
        )
        generated_text = response_openai.choices[0].message['content'].strip()
        logging.info(f"Generated answer from OpenAI: '{generated_text}'")
    except openai.OpenAIError as e:
        logging.error(f"Failed to generate response from OpenAI: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response from OpenAI: {e}")

    # Optional: Further processing or validation of the answer
    if len(generated_text.split()) < 5:
        logging.warning(f"Generated answer is too short for query '{query}'.")
        generated_text = "I'm sorry, I couldn't understand your question. Could you please rephrase?"

    # Generate a unique message ID
    message_id = str(uuid.uuid4())

    # Log the conversation with intent to MongoDB, including message_id
    await log_conversation(message_id=message_id, user_query=query, bot_response=generated_text, intent=intent)

    return ChatResponse(message_id=message_id, answer=generated_text)

@app.post("/feedback/", response_class=JSONResponse)
async def feedback(request: FeedbackRequest):
    """
    Endpoint to receive user feedback (Like/Dislike) for a specific message.
    Updates the corresponding conversation_history document with the feedback.
    """
    try:
        result = await conversation_collection.update_one(
            {"message_id": request.message_id},
            {"$set": {"feedback": request.feedback}}
        )
        if result.matched_count == 0:
            logging.warning(f"No conversation found with message_id: {request.message_id}")
            raise HTTPException(status_code=404, detail="Message ID not found.")
        logging.info(f"Feedback '{request.feedback}' recorded for message ID {request.message_id}.")
        return JSONResponse(content={"message": f"Feedback '{request.feedback}' recorded for message ID {request.message_id}."}, status_code=200)
    except HTTPException as he:
        raise he  # Re-raise HTTPExceptions
    except Exception as e:
        logging.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback.")

@app.get("/conversation-history/", response_class=JSONResponse)
async def get_conversation_history():
    """
    Endpoint to retrieve the entire conversation history from MongoDB.
    """
    try:
        conversations = []
        cursor = conversation_collection.find({})
        async for document in cursor:
            # Convert ObjectId to string and datetime to ISO format if necessary
            document['_id'] = str(document['_id'])
            document['timestamp'] = document['timestamp'].isoformat() + "Z"
            conversations.append(document)
        return JSONResponse(content=conversations, status_code=200)
    except Exception as e:
        logging.error(f"Failed to retrieve conversation history from MongoDB: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation history.")

# Removed FeedbackLog and feedback_history collection related code

# ============================
# Run the FastAPI app
# ============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Run the application

