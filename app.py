import os
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

@app.route("/api/voice", methods=["POST"])
def voice_chat():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_file.save(temp_audio.name)
        temp_path = temp_audio.name

    try:
        with open(temp_path, "rb") as f:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        user_text = transcript.text.strip()

        gpt_response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a friendly English tutor."},
                {"role": "user", "content": user_text}
            ]
        )
        reply_text = gpt_response.choices[0].message.content.strip()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_tts:
            tts_response = openai.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=reply_text
            )
            for chunk in tts_response.iter_bytes():
                temp_tts.write(chunk)
            tts_path = temp_tts.name

        return jsonify({
            "userText": user_text,
            "assistantText": reply_text,
            "assistantAudioUrl": f"/api/audio/{Path(tts_path).name}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_path)

@app.route("/api/audio/<filename>")
def get_audio(filename):
    file_path = Path(tempfile.gettempdir()) / filename
    if not file_path.exists():
        return jsonify({"error": "Audio not found"}), 404
    return send_file(file_path, mimetype="audio/mpeg")

if __name__ == "__main__":
    app.run(debug=False, port=5000, use_reloader=False)
