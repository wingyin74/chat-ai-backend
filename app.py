import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect
import openai
from openai import OpenAI
import websockets
import base64
import threading
from queue import Queue
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)

@dataclass
class TranscriptEntry:
    timestamp: str
    speaker: str  # 'user' or 'assistant'
    message: str
    message_type: str = 'text'  # 'text', 'exercise_request', 'exercise_response'

class ConversationManager:
    def __init__(self):
        self.transcripts: Dict[str, List[TranscriptEntry]] = {}
        self.active_sessions: Dict[str, Dict] = {}
    
    def add_transcript_entry(self, session_id: str, speaker: str, message: str, message_type: str = 'text'):
        if session_id not in self.transcripts:
            self.transcripts[session_id] = []
        
        entry = TranscriptEntry(
            timestamp=datetime.now().isoformat(),
            speaker=speaker,
            message=message,
            message_type=message_type
        )
        self.transcripts[session_id].append(entry)
        logger.info(f"Session {session_id}: {speaker} - {message[:100]}...")
    
    def get_transcript(self, session_id: str) -> List[Dict]:
        return [asdict(entry) for entry in self.transcripts.get(session_id, [])]
    
    def create_session(self, session_id: str):
        self.active_sessions[session_id] = {
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'realtime_connection': None
        }
    
    def close_session(self, session_id: str):
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['status'] = 'closed'
            if 'realtime_connection' in self.active_sessions[session_id]:
                connection = self.active_sessions[session_id]['realtime_connection']
                if connection and hasattr(connection, 'close'):
                    asyncio.create_task(connection.close())

conversation_manager = ConversationManager()

class ExerciseGenerator:
    """Handles agent handoff to text-based LLM for exercise creation"""
    
    @staticmethod
    def create_exercise(user_level: str = "beginner", topic: str = "general", exercise_type: str = "vocabulary"):
        """Agent handoff to text-based LLM for exercise creation"""
        try:
            prompt = f"""
            Create an English learning exercise for a {user_level} level student.
            Topic: {topic}
            Exercise type: {exercise_type}
            
            Return a JSON object with:
            {{
                "exercise_type": "{exercise_type}",
                "topic": "{topic}",
                "level": "{user_level}",
                "instructions": "Clear instructions for the voice agent",
                "content": {{
                    "word": "target word",
                    "definition": "simple definition",
                    "example_sentence": "example sentence",
                    "pronunciation_tip": "pronunciation guidance",
                    "difficulty_level": 1-5
                }},
                "interaction_flow": [
                    "Step 1: Introduce the word",
                    "Step 2: Have user repeat pronunciation",
                    "Step 3: Ask user to use in sentence",
                    "Step 4: Provide feedback"
                ]
            }}
            
            Make it engaging and appropriate for voice interaction.
            """
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            
            exercise_data = json.loads(response.choices[0].message.content)
            logger.info(f"Generated exercise: {exercise_data.get('content', {}).get('word', 'N/A')}")
            return exercise_data
            
        except Exception as e:
            logger.error(f"Exercise generation failed: {e}")
            return {
                "exercise_type": "vocabulary",
                "topic": "basic",
                "level": "beginner",
                "instructions": "Let's practice a simple word",
                "content": {
                    "word": "hello",
                    "definition": "a greeting used when meeting someone",
                    "example_sentence": "Hello, how are you today?",
                    "pronunciation_tip": "HEL-loh",
                    "difficulty_level": 1
                },
                "interaction_flow": [
                    "Introduce the word 'hello'",
                    "Have user repeat pronunciation",
                    "Ask user to use it in a sentence",
                    "Provide encouragement"
                ]
            }

class RealtimeConnection:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.websocket = None
        self.is_connected = False
        self.message_queue = Queue()
        
    async def connect(self):
        """Connect to OpenAI Realtime API"""
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        try:
            self.websocket = await websockets.connect(url, extra_headers=headers)
            self.is_connected = True
            logger.info(f"Connected to OpenAI Realtime API for session {self.session_id}")
            
            # Configure the session
            await self.configure_session()
            
            # Start listening for messages
            asyncio.create_task(self.listen_to_openai())
            
        except Exception as e:
            logger.error(f"Failed to connect to Realtime API: {e}")
            self.is_connected = False
            raise
    
    async def configure_session(self):
        """Configure the Realtime API session"""
        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": """You are an English learning chatbot with native voice capabilities. 
                
                Your role:
                1. Help users learn English through conversation
                2. Be encouraging and patient
                3. Correct pronunciation gently
                4. When you decide to create a specific exercise, say "Let me create a custom exercise for you" and I will provide exercise content
                5. Adapt to the user's level and interests
                
                Keep responses conversational and engaging. Focus on practical English usage.""",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "wav",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200
                },
                "tools": [
                    {
                        "type": "function",
                        "name": "create_exercise",
                        "description": "Create a custom English learning exercise",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "level": {"type": "string", "description": "beginner, intermediate, advanced"},
                                "topic": {"type": "string", "description": "Topic for the exercise"},
                                "exercise_type": {"type": "string", "description": "vocabulary, grammar, pronunciation"}
                            },
                            "required": ["level", "topic", "exercise_type"]
                        }
                    }
                ],
                "tool_choice": "auto"
            }
        }
        
        await self.websocket.send(json.dumps(config))
        logger.info("Configured Realtime API session")
    
    async def listen_to_openai(self):
        """Listen for messages from OpenAI Realtime API"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self.handle_openai_message(data)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"OpenAI connection closed for session {self.session_id}")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Error listening to OpenAI: {e}")
            self.is_connected = False
    
    async def handle_openai_message(self, data):
        """Handle messages from OpenAI Realtime API"""
        message_type = data.get("type")
        
        if message_type == "conversation.item.input_audio_transcription.completed":
            # User speech transcription
            transcript = data.get("transcript", "")
            conversation_manager.add_transcript_entry(
                self.session_id, "user", transcript, "text"
            )
            
            # Emit to frontend
            socketio.emit('transcript_update', {
                'session_id': self.session_id,
                'speaker': 'user',
                'message': transcript
            })
        
        elif message_type == "response.audio_transcript.done":
            # Assistant response transcription
            transcript = data.get("transcript", "")
            conversation_manager.add_transcript_entry(
                self.session_id, "assistant", transcript, "text"
            )
            
            # Emit to frontend
            socketio.emit('transcript_update', {
                'session_id': self.session_id,
                'speaker': 'assistant',
                'message': transcript
            })
        
        elif message_type == "response.function_call_arguments.done":
            # Function call for exercise creation
            function_name = data.get("name")
            if function_name == "create_exercise":
                args = json.loads(data.get("arguments", "{}"))
                exercise = ExerciseGenerator.create_exercise(
                    user_level=args.get("level", "beginner"),
                    topic=args.get("topic", "general"),
                    exercise_type=args.get("exercise_type", "vocabulary")
                )
                
                # Log exercise creation
                conversation_manager.add_transcript_entry(
                    self.session_id, "assistant", 
                    f"Created exercise: {exercise['content']['word']}", 
                    "exercise_request"
                )
                
                # Return exercise to OpenAI
                await self.send_function_result(data.get("call_id"), exercise)
        
        elif message_type == "response.audio.delta":
            # Stream WAV audio to frontend
            audio_data = data.get("delta", "")
            socketio.emit('audio_stream', {
                'session_id': self.session_id,
                'audio_data': audio_data,
                'audio_mime': 'audio/wav',
                'encoding': 'base64'
            })
        
        # Forward all messages to frontend for debugging
        socketio.emit('realtime_message', {
            'session_id': self.session_id,
            'data': data
        })
    
    async def send_function_result(self, call_id: str, result: dict):
        """Send function call result back to OpenAI"""
        message = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result)
            }
        }
        await self.websocket.send(json.dumps(message))
    
    async def send_audio(self, audio_data: str):
        """Send audio data to OpenAI"""
        if not self.is_connected:
            return
        
        message = {
            "type": "input_audio_buffer.append",
            "audio": audio_data
        }
        await self.websocket.send(json.dumps(message))
    
    async def commit_audio(self):
        """Commit audio buffer"""
        if not self.is_connected:
            return
        
        message = {"type": "input_audio_buffer.commit"}
        await self.websocket.send(json.dumps(message))
    
    async def create_response(self):
        """Trigger response generation"""
        if not self.is_connected:
            return
        
        message = {"type": "response.create"}
        await self.websocket.send(json.dumps(message))
    
    async def close(self):
        """Close the connection"""
        self.is_connected = False
        if self.websocket:
            await self.websocket.close()

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    logger.info(f"Client connected: {session_id}")
    
    conversation_manager.create_session(session_id)
    socketio.emit('connected', {'session_id': session_id})

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    logger.info(f"Client disconnected: {session_id}")
    
    conversation_manager.close_session(session_id)

@socketio.on('start_conversation')
def handle_start_conversation():
    session_id = request.sid
    logger.info(f"Starting conversation for session: {session_id}")
    
    # Create and start Realtime connection in a separate thread
    def start_realtime_connection():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def connect_and_store():
            connection = RealtimeConnection(session_id)
            await connection.connect()
            conversation_manager.active_sessions[session_id]['realtime_connection'] = connection
            
            socketio.emit('conversation_started', {'session_id': session_id})
        
        loop.run_until_complete(connect_and_store())
        loop.run_forever()
    
    thread = threading.Thread(target=start_realtime_connection)
    thread.daemon = True
    thread.start()

@socketio.on('send_audio')
def handle_send_audio(data):
    session_id = request.sid
    audio_data = data.get('audio_data', '')
    
    if session_id in conversation_manager.active_sessions:
        connection = conversation_manager.active_sessions[session_id].get('realtime_connection')
        if connection and connection.is_connected:
            # Send audio in a separate thread to avoid blocking
            def send_audio_async():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(connection.send_audio(audio_data))
            
            thread = threading.Thread(target=send_audio_async)
            thread.daemon = True
            thread.start()

@socketio.on('commit_audio')
def handle_commit_audio():
    session_id = request.sid
    
    if session_id in conversation_manager.active_sessions:
        connection = conversation_manager.active_sessions[session_id].get('realtime_connection')
        if connection and connection.is_connected:
            def commit_audio_async():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(connection.commit_audio())
                loop.run_until_complete(connection.create_response())
            
            thread = threading.Thread(target=commit_audio_async)
            thread.daemon = True
            thread.start()

# REST API endpoints
@app.route('/api/transcript/<session_id>', methods=['GET'])
def get_transcript(session_id):
    """Get full transcript for a session"""
    transcript = conversation_manager.get_transcript(session_id)
    return jsonify({
        'session_id': session_id,
        'transcript': transcript,
        'total_entries': len(transcript)
    })

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all active sessions"""
    return jsonify({
        'active_sessions': list(conversation_manager.active_sessions.keys()),
        'total_sessions': len(conversation_manager.active_sessions)
    })

@app.route('/api/exercise', methods=['POST'])
def create_manual_exercise():
    """Manually create an exercise (for testing)"""
    data = request.json
    exercise = ExerciseGenerator.create_exercise(
        user_level=data.get('level', 'beginner'),
        topic=data.get('topic', 'general'),
        exercise_type=data.get('exercise_type', 'vocabulary')
    )
    return jsonify(exercise)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(conversation_manager.active_sessions)
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask app on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=debug)