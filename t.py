import os
import cv2
import numpy as np
import base64
import logging
import google.generativeai as genai
from flask import Flask, request, render_template_string, redirect, session, url_for
from templates import HTML_CAMERA, HTML_QUIZ, HTML_CHAT, HTML_RESULT


app = Flask(__name__)
app.secret_key = "mental_health_secure_key_2024" 


GEMINI_API_KEY = "AIzaSyC1lLjIKcLe6hI4gpdZ3haRWfSJy0rm2h4"

# --- AI SERVICES SETUP ---
LLM_AVAILABLE = False
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    LLM_AVAILABLE = True
    print("✅ Gemini AI Connected.")
except Exception as e:
    print(f"⚠️ Gemini Error: {e}")

VISION_AVAILABLE = False
try:
    from deepface import DeepFace
    print("⏳ Loading Face AI...")
    DeepFace.build_model("Emotion")
    VISION_AVAILABLE = True
    print("✅ DeepFace Loaded.")
except Exception as e:
    print(f"❌ DeepFace Error: {e}")

# --- UPDATED 7 QUESTIONS ---
QUESTIONS = {
    1: {"text": "How have you been sleeping lately?", "options": [("Great (7+ hours)", 2), ("Okay / Average", 1), ("Poorly / Insomnia", 0)]},
    2: {"text": "How are your energy levels today?", "options": [("High / Energetic", 2), ("Moderate", 1), ("Low / Exhausted", 0)]},
    3: {"text": "Have you felt anxious or restless recently?", "options": [("Not at all", 2), ("A little bit", 1), ("Very much", 0)]},
    4: {"text": "How easily can you focus on tasks?", "options": [("Sharp / Focused", 2), ("Slightly distracted", 1), ("Complete brain fog", 0)]},
    5: {"text": "How is your appetite lately?", "options": [("Normal/Healthy", 2), ("Eating too much/little", 1), ("No appetite at all", 0)]},
    6: {"text": "How connected do you feel to friends/family?", "options": [("Very connected", 2), ("A bit isolated", 1), ("Alone/Lonely", 0)]},
    7: {"text": "How hopeful do you feel about the future?", "options": [("Optimistic", 2), ("Uncertain", 1), ("Struggling to see hope", 0)]},
}

# --- CORE LOGIC ---

def analyze_face(base64_str):
    if not VISION_AVAILABLE or not base64_str:
        return "Neutral", 0
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        np_arr = np.frombuffer(base64.b64decode(base64_str), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        analysis = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False, silent=True)
        raw_emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
        return raw_emotion, 1
    except:
        return "Neutral", 0

def map_emotion_to_therapy(raw_emotion):
    mapping = {"happy": "Radiant", "sad": "Heavy-hearted", "angry": "Frustrated", 
               "fear": "Anxious", "surprise": "Unsettled", "neutral": "Calm", "disgust": "Uncomfortable"}
    return mapping.get(raw_emotion.lower(), raw_emotion.capitalize())

def get_therapist_response(user_input, history, mood_data):
    if not LLM_AVAILABLE:
        return "I'm having trouble connecting to my brain. Please check the API key."
    
    # SYSTEM PROMPT: Defines the AI's personality
    system_instruction = (
        f"You are a kind, empathetic AI Therapist. The patient is currently feeling {mood_data['face_desc']}. "
        f"Their wellness score is {mood_data['score']}/14. "
        "Keep responses under 3 sentences. Be supportive, validate their feelings, "
        "and always end with a question that encourages them to share more."
    )

    # Convert session history to Gemini format
    messages = []
    for sender, msg in history:
        role = "user" if sender == "You" else "model"
        messages.append({"role": role, "parts": [msg]})

    try:
        chat = model.start_chat(history=messages[:-1]) # Start chat with previous history
        response = chat.send_message(f"SYSTEM NOTE: {system_instruction}\n\nUSER SAYS: {user_input}")
        return response.text
    except Exception as e:
        return f"I'm listening, but I had a technical hiccup: {str(e)}"

# --- ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        session.clear()
        session['raw_face'], _ = analyze_face(request.form.get('image_data'))
        session['answers'] = []
        return redirect(url_for('quiz', q_id=1))
    return render_template_string(HTML_CAMERA)

@app.route('/quiz/<int:q_id>', methods=['GET', 'POST'])
def quiz(q_id):
    if q_id > len(QUESTIONS):
        return redirect(url_for('results'))
    if request.method == 'POST':
        ans = int(request.form['answer'])
        answers = session.get('answers', [])
        answers.append(ans)
        session['answers'] = answers
        return redirect(url_for('quiz', q_id=q_id+1))
    return render_template_string(HTML_QUIZ, q=QUESTIONS[q_id], q_num=q_id, total=len(QUESTIONS))

@app.route('/results')
def results():
    answers = session.get('answers', [])
    score = sum(answers)
    raw_face = session.get('raw_face', 'neutral')
    
    # Scoring Logic (Max score is now 14)
    if score >= 10: status, color = "High Well-being", "#2ed573"
    elif score >= 6: status, color = "Moderate Well-being", "#ffa502"
    else: status, color = "Needs Extra Care", "#ff4757"

    session['analysis'] = {'raw_face': raw_face, 'face_desc': map_emotion_to_therapy(raw_face), 
                          'score': score, 'mood_label': status, 'color': color}
    session['history'] = []
    return render_template_string(HTML_RESULT, data=session['analysis'])

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    data = session.get('analysis')
    history = session.get('history', [])
    if request.method == 'POST':
        msg = request.form['message']
        history.append(('You', msg))
        ai_reply = get_therapist_response(msg, history, data)
        history.append(('Therapist', ai_reply))
        session['history'] = history
        return redirect(url_for('chat'))
    return render_template_string(HTML_CHAT, history=history, data=data)

# (Keep the HTML Templates and CSS from your original code)
# Note: Ensure HTML_CHAT and HTML_QUIZ are included below the routes.