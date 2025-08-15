import os
import uuid # For generating unique session IDs
import time
import fitz # Library PyMuPDF for PDF processing
import threading # For running cleanup thread
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from groq import Groq
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rich import _console

app = Flask(__name__)
CORS(app)
import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# Initialize Llama3 API client

# Configuration
ALLOWED_EXTENSIONS = {'pdf'} # Only PDF files are allowed for now

# Intent Keywords
MCQ_KEYWORDS = [
    'mcq', 'multiple choice', 'quiz', 'create quiz', 'create mcqs', 'generate mcqs',
    'multiple choice questions', 'objective questions', 'make quiz', 'create test',
    'generate quiz', 'make mcqs', 'create multiple choice', 'test questions',
    'objective test', 'generate test', 'create assessment'
]

SUMMARY_KEYWORDS = [
    'summary', 'summarize', 'summarise', 'brief', 'synopsis',
    'give me summary', 'create summary', 'generate summary',
    'provide summary', 'make summary', 'brief overview',
    'short version', 'key points', 'main points', 'highlight',
    'gist', 'outline', 'summarize text', 'brief summary'
]

QUESTION_KEYWORDS = [
    'generate questions', 'create questions', 'make questions', 'questions only',
    'ask questions', 'give me questions', 'generate some questions',
    'create practice questions', 'make test questions', 'prepare questions',
    'write questions', 'form questions', 'question bank', 'practice questions',
    'sample questions', 'test questions only'
]

QUESTION_ANSWER_KEYWORDS = [
    'generate questions and answers', 'create questions and answers',
    'make questions and answers', 'short questions and answers',
    'q and a', 'q&a', 'questions with answers', 'practice q&a',
    'generate q&a', 'create q&a', 'make q&a', 'questions & answers',
    'questions answers both', 'complete questions answers',
    'full questions and answers', 'detailed questions and answers'
]

class ChatSession:
    def __init__(self, max_messages=10):
        self.messages = []
        self.last_accessed = datetime.now()
        self.max_messages = max_messages
        self.document_context = None

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the session and maintain message limit."""
        message = {"role": role, "content": content}
        self.messages.append(message)
       
        # Keep only the last max_messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
       
        self.last_accessed = datetime.now()

    def get_recent_context(self, num_messages=5) -> List[Dict[str, str]]:
        """Get the most recent messages for context."""
        return self.messages[-num_messages:] if self.messages else []

class DocumentStorage: # Class to store the current document and related data
    def __init__(self):
        self.current_document: str = None
        self.last_uploaded: datetime = None
        self.file_path: str = None

document_storage = DocumentStorage() # Initialize document storage
chat_sessions: Dict[str, ChatSession] = {} # Initialize chat sessions

# To check if a file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# To extract text from a PDF file
def extract_text_from_pdf(file_path_or_stream):
    doc = fitz.open(stream=file_path_or_stream.read(), filetype="pdf") if hasattr(file_path_or_stream, 'read') else fitz.open(file_path_or_stream)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text
# Generate response from the model
def generate_llama_response(conversation_history: List[Dict[str, str]], prompt, system_role="You are a helpful assistant."):
    try:
        # Call the Llama3 API to generate a response
        response = client.chat.completions.create(
            messages= conversation_history + [
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            max_tokens=8192,
            temperature=0.7 # Higher temperature means more randomness
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return None
# Extract question count from message
def extract_question_count(message):
    import re
    match = re.search(r'(\d+)\s*(?:questions?|mcqs?|quiz(?:zes)?)', message.lower())
    return int(match.group(1)) if match else 5

# Extract line count from message
def extract_line_count(message):
    import re
    match = re.search(r'(\d+)\s*lines?', message.lower())
    return int(match.group(1)) if match else None

# Generate multiple choice questions
def generate_mcq(document_text, num_questions=5):
    prompt = f"""Generate {num_questions} multiple choice questions based on the following text. For each question:
    - Provide a clear question
    - Provide 4 plausible options (A, B, C, D)
    - Mark the correct answer with an asterisk (*)
    Document Text: {document_text[:2000]}"""
    return generate_llama_response(prompt, "You are an expert at generating multiple-choice questions from text.")

# Generate summary of text
def generate_summary(text, num_lines=None):
    try:
        prompt = f"Summarize the following text"
        if num_lines:
            prompt += f" in exactly {num_lines} lines"
        prompt += f":\n\n{text}"
       
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        # Choosing first choice if available or returning error message
        return response.choices[0].message.content if response.choices else "Could not generate summary."
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Generate questions only
def generate_questions_only(document_text, num_questions=5):
    prompt = f"""Generate {num_questions} questions based on the following text.
    Format: Number each question (1., 2., etc.)
    Document Text: {document_text[:2000]}"""
    return generate_llama_response(prompt, "You are an expert at generating questions from text.")

# Generate questions and answers
def generate_questions_and_answers(document_text, num_questions=5):
    prompt = f"""Generate {num_questions} questions and their answers based on the following text.
    Format each Q&A pair as:
    Question 1: [question text]
    Answer 1: [answer text]
   
    Document Text: {document_text[:2000]}"""
    return generate_llama_response(prompt, "You are an expert at generating questions and answers from text.")

def classify_intent(message: str) -> str:
    prompt = f"""Classify the following message into exactly one of these categories:
    - summary: If asking for a summary or synopsis
    - mcq: If requesting multiple choice questions
    - questions_only: If asking for just questions without answers
    - questions_and_answers: If asking for both questions and answers
    - document interaction: If asking a question based on the document
   
    Message: {message}
   
    Return only the category name, nothing else."""
   
    try:
        response = generate_llama_response(
            prompt=prompt,
            system_role="You are an expert at classifying user intents. Respond with exactly one category name."
        )
        return response.strip().lower()
    except Exception as e:
        print(f"Error in intent classification: {e}")
        return "document interaction"

def process_with_intent(message: str, document_text: str = None) -> str:
    try:
        intent = classify_intent(message)
        print(f"Intent classified as: {intent}")
       
        num_questions = extract_question_count(message) if 'question' in intent or intent == 'mcq' else 5
        num_lines = extract_line_count(message) if intent == 'summary' else None
       
        if not document_text:
            return "No document has been uploaded yet."
           
        if intent == 'summary':
            return generate_summary(document_text, num_lines)
        elif intent == 'mcq':
            return generate_mcq(document_text, num_questions)
        elif intent == 'questions_only':
            return generate_questions_only(document_text, num_questions)
        elif intent == 'questions_and_answers':
            return generate_questions_and_answers(document_text, num_questions)
        elif intent == 'document interaction':
            return process_document_query()
        else:
            return "Document is uploaded if you want to ask normal questions please remove the document."
           
    except Exception as e:
        print(f"Error in intent processing: {e}")
        return f"An error occurred while processing your request: {str(e)}"
# Processing all the queries with document
def process_document_query(question: str, document_text: str, session_id: str) -> dict:
    try:
        if not document_text:
            return {"error": "No document available", "success": False}

        # Get the session and its message history
        session = chat_sessions.get(session_id)
        if not session:
            session = ChatSession()
            chat_sessions[session_id] = session

        # Create messages array including conversation history
        messages = [
            {"role": "system", "content": "You are a helpful assistant analyzing a document. Maintain context of the conversation."},
            {"role": "user", "content": f"Document Context:\n{document_text[:2000]}\n\nRemember this document for our conversation."},
            {"role": "assistant", "content": "I understand the document and will maintain context of our conversation."}
        ]
       
        # Add recent conversation history (last 5 messages)
        messages.extend(session.messages[-5:])
       
        # Add the current question
        messages.append({"role": "user", "content": question})

        # Get response from the model
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages
        )
       
        answer = str(response.choices[0].message.content) if response.choices else 'Could not generate an answer.'
       
        # Update session with the new interaction
        session.add_message("user", question)
        session.add_message("assistant", answer)
       
        return {
            "answer": answer,
            "session_id": session_id,
            "success": True
        }
    except Exception as e:
        print(f"Error in document query: {str(e)}")
        return {
            "error": str(e),
            "session_id": session_id,
            "success": False
        }
# Cleanup old sessions
def cleanup_old_sessions():
    while True:
        try:
            current_time = datetime.now()
            expired_sessions = [
                session_id for session_id, session in chat_sessions.items()
                if current_time - session.last_accessed > timedelta(hours=1)
            ]
            for session_id in expired_sessions:
                del chat_sessions[session_id]
        except Exception as e:
            print(f"Error in cleanup thread: {str(e)}")
        time.sleep(300)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-document', methods=['POST'])
def upload_document():
    try:
        if 'document' not in request.files:
            return jsonify({"error": "No document uploaded"}), 400
       
        document = request.files['document']
        if document and allowed_file(document.filename):
            document_content = extract_text_from_pdf(document)
            document_storage.current_document = document_content
            document_storage.last_uploaded = datetime.now()
            return jsonify({"message": "Document uploaded successfully"}), 200
       
        return jsonify({"error": "Invalid file type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '').lower()
        session_id = data.get('session_id', str(uuid.uuid4()))
       
        # Get or create session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = ChatSession()
        session = chat_sessions[session_id]
        session.last_accessed = datetime.now()

        if document_storage.current_document:
            # Process with document context
            result = process_document_query(message, document_storage.current_document, session_id)
            return jsonify(result), 200 if result["success"] else 500
           
        # For non-document conversations
        messages = session.get_recent_context()
        response = generate_llama_response(messages, message)
       
        # Update session
        session.add_message("user", message)
        session.add_message("assistant", response)
       
        return jsonify({
            "answer": response,
            "session_id": session_id,
            "success": True
        }), 200
       
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "session_id": session_id if 'session_id' in locals() else None,
            "success": False
        }), 500
# Clear chat history
@app.route('/clear-chat', methods=['POST'])
def clear_chat():
    print('clear chat clicked')
    try:
        data = request.get_json()
        if not data:
            print("No JSON data received")
            return jsonify({
                "error": "No data provided",
                "success": False
            }), 400
           
        session_id = data.get('session_id')
        if not session_id:
            print("No session ID in request")
            return jsonify({
                "error": "No session ID provided",
                "success": False
            }), 400

        print(f"Attempting to clear chat for session: {session_id}")
       
        if session_id in chat_sessions:
            print(f"Clearing existing session: {session_id}")
            chat_sessions[session_id].messages = []
            chat_sessions[session_id].last_accessed = datetime.now()
        else:
            print(f"Creating new session: {session_id}")
            chat_sessions[session_id] = ChatSession()
        # Clearing document storage  
        if hasattr(document_storage, 'current_document'):
            document_storage.current_document = None
            # Clearing file path
            if hasattr(document_storage, 'file_path') and document_storage.file_path:
                try:
                    if os.path.exists(document_storage.file_path):
                        os.remove(document_storage.file_path)
                    document_storage.file_path = None
                except Exception as e:
                    print(f"Warning: Could not remove file: {e}")
            print("Cleared document storage")

        return jsonify({
            "message": "Chat history cleared successfully",
            "session_id": session_id,
            "success": True
        }), 200
       
    except Exception as e:
        print(f"Error in clear_chat: {str(e)}")
        return jsonify({
            "error": f"Failed to clear chat: {str(e)}",
            "success": False
        }), 500

# Discard the current document
@app.route('/discard-document', methods=['POST'])
def discard_document():
    print('discard document clicked')
    try:
        document_storage.current_document = None
        document_storage.last_uploaded = None
       
        if document_storage.file_path and os.path.exists(document_storage.file_path):
            os.remove(document_storage.file_path)
       
        document_storage.file_path = None
       
        return jsonify({
            "message": "Document discarded successfully",
            "success": True
        }), 200

    except Exception as e:
        print(f"Error in discard_document: {str(e)}")
        return jsonify({
            "error": f"Failed to discard document: {str(e)}",
            "success": False
        }), 500
   
chat_sessions: Dict[str, ChatSession] = {}

if __name__ == '__main__':
    cleanup_thread = threading.Thread(target=cleanup_old_sessions, daemon=True)
    cleanup_thread.start()
    app.run(debug=True)