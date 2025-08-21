# Chat AI Backend

A real-time English learning chatbot backend built with Flask, Socket.IO, and OpenAI's Realtime API.

## Quick Setup

1. **Install Python 3.8+ and clone the repo**
   ```bash
   git clone <your-repo-url>
   cd chat-ai-backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file**
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   SECRET_KEY=your_secret_key_here
   PORT=5000
   FLASK_ENV=development
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

## Requirements

- OpenAI API key with GPT-4 and Realtime API access
- Python 3.12+
- Audio recording capabilities

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/sessions` - Active sessions
- `GET /api/transcript/<session_id>` - Get transcript
- `POST /api/exercise` - Create exercise

## WebSocket Events

- `start_conversation` - Initialize conversation
- `send_audio` - Send audio data
- `commit_audio` - Generate AI response

The server will be available at `http://localhost:5000` 