from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

import assemblyai as aai
import os
import base64
import requests
import tempfile
import json
import uuid


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")


# Check API keys
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in .env")

if not MURF_API_KEY:
    raise ValueError("MURF_API_KEY is missing in .env")

if not ASSEMBLYAI_API_KEY:
    raise ValueError("ASSEMBLYAI_API_KEY is missing in .env")


# AssemblyAI configuration
aai.settings.api_key = ASSEMBLYAI_API_KEY


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

CORS(
    app,
    expose_headers=[
        "X-Question-Number",
        "X-Interview-Complete"
    ]
)


# ============================================================
# AI MODEL
# ============================================================

model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key=GOOGLE_API_KEY
)


# ============================================================
# GLOBAL INTERVIEW STATE
# ============================================================

checkpointer = None
agent = None

question_count = 0
current_subject = ""
thread_id = None


# ============================================================
# INTERVIEW PROMPT
# ============================================================

INTERVIEW_PROMPT = """
You are Natalie, a friendly and conversational AI interviewer conducting
a natural {subject} interview.

IMPORTANT GUIDELINES:

1. Ask exactly 5 questions total throughout the interview.
2. Keep questions SHORT and CRISP.
3. Questions should normally be 1-2 sentences maximum.
4. ALWAYS reference what the candidate ACTUALLY said in their previous answer.
5. NEVER make up or assume information about the candidate.
6. Show genuine interest with a short acknowledgment.
7. Adapt the next question based on the candidate's actual answer.
8. If the candidate gives a strong answer, ask a slightly deeper question.
9. If the candidate is confused or says "I don't know", ask a simpler question.
10. Be warm, professional and conversational.
11. Do not give lengthy explanations.
12. Focus on asking one clear question at a time.

CRITICAL:
Read the conversation history carefully.
Only reference information that the candidate actually provided.

Keep the interview short, natural and adaptive.
"""


# ============================================================
# FEEDBACK PROMPT
# ============================================================

FEEDBACK_PROMPT = """
Based on our complete interview conversation, provide detailed feedback.

Return ONLY valid JSON.

{
    "subject": "<topic>",
    "candidate_score": <number from 1 to 5>,
    "feedback": "<specific strengths with examples from the candidate's actual answers>",
    "areas_of_improvement": "<specific constructive suggestions based on actual weaknesses or gaps>"
}

IMPORTANT:
- Do not invent information.
- Use only the candidate's actual answers.
- Mention specific strengths.
- Mention specific areas that need improvement.
- Give a fair score from 1 to 5.
- Return JSON only.
"""


# ============================================================
# MURF TEXT TO SPEECH
# ============================================================

def stream_audio(text):
    """
    Convert AI-generated text into speech using Murf.
    Returns base64 encoded MP3 chunks.
    """

    BASE_URL = "https://global.api.murf.ai/v1/speech/stream"

    payload = {
        "text": text,
        "voiceId": "en-US-natalie",
        "model": "FALCON",
        "locale": "en-US",
        "sampleRate": 24000,
        "format": "MP3"
    }

    headers = {
        "Content-Type": "application/json",
        "api-key": MURF_API_KEY
    }

    try:
        response = requests.post(
            BASE_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=60
        )

        if response.status_code != 200:
            print(
                "Murf API Error:",
                response.status_code,
                response.text
            )

            raise Exception(
                f"Murf API request failed: {response.status_code}"
            )

        for chunk in response.iter_content(chunk_size=4096):

            if chunk:
                encoded_chunk = base64.b64encode(chunk).decode("utf-8")

                yield encoded_chunk + "\n"

    except Exception as error:
        print("Murf error:", error)
        raise


# ============================================================
# CREATE NEW INTERVIEW AGENT
# ============================================================

def create_interview_agent():
    """
    Create a fresh LangGraph agent and memory
    for every new interview.
    """

    global checkpointer
    global agent

    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer
    )


# ============================================================
# SPEECH TO TEXT - ASSEMBLYAI
# ============================================================

def speech_to_text(audio_path):

    print("AssemblyAI transcription started...")

    try:

        transcriber = aai.Transcriber()

        config = aai.TranscriptionConfig(
            speech_models=[
                "universal-3-pro",
                "universal-2"
            ],
            language_detection=True,
            speaker_labels=False
        )

        transcript = transcriber.transcribe(
            audio_path,
            config=config
        )

        print(
            "AssemblyAI status:",
            transcript.status
        )

        if transcript.status == aai.TranscriptStatus.error:

            print(
                "AssemblyAI error:",
                transcript.error
            )

            return ""

        text = transcript.text or ""

        print(
            "Transcribed text:",
            text
        )

        return text.strip()

    except Exception as error:

        print(
            "AssemblyAI exception:",
            error
        )

        return ""


# ============================================================
# START INTERVIEW
# ============================================================

@app.route("/start-interview", methods=["POST"])
def start_interview():

    global question_count
    global current_subject
    global thread_id

    try:

        data = request.get_json() or {}

        current_subject = data.get(
            "subject",
            "Python"
        )

        question_count = 1

        # Create unique interview session
        thread_id = str(uuid.uuid4())

        # Create fresh agent and memory
        create_interview_agent()

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        formatted_prompt = INTERVIEW_PROMPT.format(
            subject=current_subject
        )

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": formatted_prompt
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Start the interview with a warm greeting "
                            f"and ask the first question about "
                            f"{current_subject}. "
                            f"Keep it SHORT."
                        )
                    }
                ]
            },
            config=config
        )

        question = response["messages"][-1].text

        print()
        print("=" * 60)
        print(f"Interview Started")
        print(f"Subject: {current_subject}")
        print(f"Question {question_count}: {question}")
        print("=" * 60)

        return Response(
            stream_audio(question),
            mimetype="text/plain"
        )

    except Exception as error:

        print("Start interview error:", error)

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# ============================================================
# SUBMIT ANSWER
# ============================================================

@app.route("/submit-answer", methods=["POST"])
def submit_answer():

    global question_count

    temp_path = None

    try:

        print("\n" + "=" * 60)
        print("SUBMIT ANSWER REQUEST RECEIVED")
        print("=" * 60)

        # ----------------------------------------------------
        # 1. Check audio
        # ----------------------------------------------------

        if "audio" not in request.files:

            print("ERROR: No audio file received")

            return jsonify({
                "success": False,
                "error": "No audio file received"
            }), 400

        audio_file = request.files["audio"]

        print("Audio received:", audio_file.filename)

        # ----------------------------------------------------
        # 2. Save temporary audio
        # ----------------------------------------------------

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".webm"
        )

        temp_path = temp_file.name
        temp_file.close()

        audio_file.save(temp_path)

        print("Audio saved:", temp_path)

        # ----------------------------------------------------
        # 3. AssemblyAI - Speech to Text
        # ----------------------------------------------------

        print("Sending audio to AssemblyAI...")

        answer = speech_to_text(temp_path)

        print("AssemblyAI result:", answer)

        # Delete temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

        temp_path = None

        # ----------------------------------------------------
        # 4. Handle empty transcription
        # ----------------------------------------------------

        if not answer or answer.strip() == "":

            answer = (
                "[Candidate's speech could not be transcribed]"
            )

        print(f"\nCandidate Answer:")
        print(answer)

        # ----------------------------------------------------
        # 5. Check interview session
        # ----------------------------------------------------

        if agent is None:

            print("ERROR: Agent is None")

            return jsonify({
                "success": False,
                "error": "Interview agent is not initialized"
            }), 400

        if thread_id is None:

            print("ERROR: Thread ID is None")

            return jsonify({
                "success": False,
                "error": "Interview session is not initialized"
            }), 400

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        # ----------------------------------------------------
        # 6. Store candidate answer in LangGraph memory
        # ----------------------------------------------------

        print("Saving candidate answer to conversation...")

        agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": answer
                    }
                ]
            },
            config=config
        )

        print("Candidate answer saved.")

        # ----------------------------------------------------
        # 7. Check if this was Question 5
        # ----------------------------------------------------

        if question_count >= 5:

            print("Question 5 completed.")

            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": """
The candidate has completed the fifth and final
interview question.

Briefly acknowledge their actual answer and tell
them that the interview is complete.

Do not ask another question.

Keep the response short and friendly.
"""
                        }
                    ]
                },
                config=config
            )

            closing_message = response[
                "messages"
            ][-1].text

            print("\nClosing message:")
            print(closing_message)

            return Response(
                stream_audio(closing_message),
                mimetype="text/plain",
                headers={
                    "X-Interview-Complete": "true"
                }
            )

        # ----------------------------------------------------
        # 8. Increment question number
        # ----------------------------------------------------

        previous_question = question_count

        question_count += 1

        print(
            f"\nGenerating Question {question_count}..."
        )

        # ----------------------------------------------------
        # 9. Generate next question using Gemini
        # ----------------------------------------------------

        next_question_prompt = f"""
The candidate has just answered Question {previous_question}.

Their actual answer is:

"{answer}"

Now continue the interview.

Generate Question {question_count} of 5.

IMPORTANT:

1. Briefly acknowledge something the candidate ACTUALLY said.
2. Do not invent information.
3. Ask exactly ONE question.
4. The new question should build naturally on their answer.
5. If their answer was strong, ask something slightly deeper.
6. If they were unsure or said "I don't know", ask something simpler.
7. Keep the entire response under 3 sentences.
8. Do not give the answer yourself.
9. Do not ask multiple questions.

Return ONLY the conversational interviewer response.
"""

        print("Sending prompt to Gemini...")

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": next_question_prompt
                    }
                ]
            },
            config=config
        )

        question = response[
            "messages"
        ][-1].text

        print("\n" + "=" * 60)
        print(f"QUESTION {question_count}")
        print(question)
        print("=" * 60)

        # ----------------------------------------------------
        # 10. Send Question to Murf
        # ----------------------------------------------------

        print(
            f"Sending Question {question_count} to Murf..."
        )

        return Response(
            stream_audio(question),
            mimetype="text/plain",
            headers={
                "X-Question-Number": str(question_count),
                "X-Interview-Complete": "false"
            }
        )

    except Exception as error:

        print("\n" + "=" * 60)
        print("SUBMIT ANSWER ERROR")
        print("=" * 60)
        print(error)
        print("=" * 60)

        if temp_path and os.path.exists(temp_path):

            try:
                os.unlink(temp_path)
            except:
                pass

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500




# ============================================================
# GET AI FEEDBACK
# ============================================================

@app.route("/get-feedback", methods=["POST"])
def get_feedback():

    try:

        if agent is None or thread_id is None:

            return jsonify({
                "success": False,
                "error": "No active interview session"
            }), 400

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        prompt = (
            FEEDBACK_PROMPT
            + "\n\n"
            + f"Interview subject: {current_subject}\n"
            + "Review the complete interview conversation."
        )

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            config=config
        )

        text = response[
            "messages"
        ][-1].text

        print()
        print("=" * 60)
        print("[AI FEEDBACK]")
        print(text)
        print("=" * 60)

        # Clean markdown JSON if Gemini adds it
        cleaned = text.strip()

        if "```json" in cleaned:

            cleaned = (
                cleaned
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        elif "```" in cleaned:

            cleaned = (
                cleaned
                .replace("```", "")
                .strip()
            )

        # Parse JSON
        try:

            feedback = json.loads(cleaned)

        except json.JSONDecodeError:

            # Try extracting JSON object
            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start != -1 and end != -1:

                json_text = cleaned[
                    start:end + 1
                ]

                feedback = json.loads(
                    json_text
                )

            else:

                raise ValueError(
                    "AI returned invalid JSON feedback"
                )

        return jsonify({
            "success": True,
            "feedback": feedback
        })

    except Exception as error:

        print(
            "Feedback error:",
            error
        )

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "message": "AI Interview Assistant backend is running"
    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("AI INTERVIEW ASSISTANT")
    print("=" * 60)
    print("Backend: http://127.0.0.1:5000")
    print("Health:  http://127.0.0.1:5000/health")
    print("=" * 60)

    app.run(
        debug=True,
        port=5000
    )