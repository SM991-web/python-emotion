import os
import cv2
import numpy as np
import base64
import logging
from flask import Flask, request, render_template_string, redirect, session, url_for

# --- CONFIGURATION ---
app = Flask(__name__)
app.secret_key = "my_super_secure_secret_key"  # Change for production

# SET YOUR API KEY HERE
# Replace the whole os.getenv part with just your key string
GEMINI_API_KEY = "AIzaSyC1lLjIKcLe6hI4gpdZ3haRWfSJy0rm2h4"

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- AI SERVICES ---

# 1. SETUP GEMINI
LLM_AVAILABLE = False
try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    LLM_AVAILABLE = True
    print("✅ Gemini AI Connected Successfully.")
except Exception as e:
    print(f"⚠️ Gemini Warning: {e}")

# 2. SETUP DEEPFACE (The Vision AI)
VISION_AVAILABLE = False
try:
    from deepface import DeepFace
    # We trigger a dummy analysis on startup to load the heavy weights immediately
    # This prevents the first user request from being slow.
    print("⏳ Loading Face AI models... (This might take 10 seconds)")
    try:
        DeepFace.build_model("Emotion")
        VISION_AVAILABLE = True
        print("✅ DeepFace AI Loaded Successfully.")
    except:
        print("⚠️ DeepFace model failed to build. Check tensorflow/keras installation.")
except ImportError:
    print("❌ DeepFace library not found. Run: pip install deepface tf-keras")

# --- CORE LOGIC ---

QUESTIONS = {
    1: {"text": "How have you been sleeping lately?", "options": [("Great (7+ hours)", 2), ("Okay / Average", 1), ("Poorly / Insomnia", 0)]},
    2: {"text": "How are your energy levels?", "options": [("High / Energetic", 2), ("Moderate", 1), ("Low / Exhausted", 0)]},
    3: {"text": "Have you felt anxious today?", "options": [("No, I'm calm", 2), ("A little bit", 1), ("Very anxious", 0)]},
    4: {"text": "How is your focus?", "options": [("Sharp / Focused", 2), ("Distracted", 1), ("Brain fog", 0)]},
}

def analyze_face(base64_str):
    """
    Robust face analysis that refuses to fail.
    Returns: A tuple (Emotion_String, Confidence_Score)
    """
    if not VISION_AVAILABLE:
        return "Simulation", 0

    try:
        # 1. Decode Image
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        
        np_arr = np.frombuffer(base64.b64decode(base64_str), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return "Error: Bad Image", 0

        # 2. Run Analysis with enforce_detection=False
        # This is the KEY FIX: It forces the AI to guess even if the face isn't perfect.
        print("📸 Analyzing image...")
        analysis = DeepFace.analyze(
            img_path=img,
            actions=['emotion'],
            detector_backend='opencv', # Fast backend
            enforce_detection=False,   # CRITICAL: Prevents crashing if face is obscure
            silent=True
        )

        # 3. Extract Result
        if isinstance(analysis, list):
            result = analysis[0]
        else:
            result = analysis

        raw_emotion = result['dominant_emotion']
        print(f"✅ Detected Raw Emotion: {raw_emotion}")
        
        return raw_emotion, 1

    except Exception as e:
        print(f"❌ Analysis Error: {e}")
        return "Neutral (Fallback)", 0

def map_emotion_to_therapy(raw_emotion):
    """
    Translates clinical AI terms into therapeutic context.
    """
    mapping = {
        "happy": "Radiant & Positive",
        "sad": "Down or Heavy-hearted",
        "angry": "Frustrated or Upset",
        "fear": "Anxious or Overwhelmed",
        "surprise": "Shocked or Unsettled",
        "neutral": "Calm / Composed",
        "disgust": "Uncomfortable"
    }
    return mapping.get(raw_emotion.lower(), raw_emotion.capitalize())

def get_therapist_response(user_input, history, mood_data):
    if not LLM_AVAILABLE:
        return "I am listening (Simulation Mode). Please check API Key."
    
    # IMPROVED SYSTEM PROMPT
    system_prompt = f"""
    You are an expert, empathetic AI Therapist.
    
    CURRENT PATIENT CONTEXT:
    1. Facial Scan: The patient appears {mood_data['face_desc']} ({mood_data['raw_face']}).
    2. Quiz Score: {mood_data['score']}/8.
    3. Overall Mood: {mood_data['mood_label']}.
    
    INSTRUCTIONS:
    - If the face is sad/anxious but they say they are fine, ask about the mismatch (masking).
    - Keep responses warm, professional, and under 3 sentences.
    - Always end with a gentle, open-ended question.
    """
    
    # Build History
    gemini_hist = [{'role': 'user', 'parts': [system_prompt]}, 
                   {'role': 'model', 'parts': ["Understood. I am ready."]}]
    
    for sender, msg in history[-6:]:
        role = 'user' if sender == 'You' else 'model'
        gemini_hist.append({'role': role, 'parts': [msg]})
        
    try:
        chat = model.start_chat(history=gemini_hist)
        response = chat.send_message(user_input)
        return response.text
    except Exception as e:
        return "I'm having a moment of silence (Connection Error). Please try again."

# --- ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        img_data = request.form.get('image_data')
        
        # Immediate Analysis
        raw_emotion, _ = analyze_face(img_data)
        
        session.clear()
        session['raw_face'] = raw_emotion
        session['answers'] = []
        
        return redirect(url_for('quiz', q_id=1))
        
    return render_template_string(HTML_CAMERA)

@app.route('/quiz/<int:q_id>', methods=['GET', 'POST'])
def quiz(q_id):
    if q_id not in QUESTIONS:
        return redirect(url_for('results'))
        
    if request.method == 'POST':
        ans = int(request.form['answer'])
        session['answers'] = session.get('answers', []) + [ans]
        return redirect(url_for('quiz', q_id=q_id+1))
        
    return render_template_string(HTML_QUIZ, q=QUESTIONS[q_id], q_num=q_id, total=len(QUESTIONS))

@app.route('/results')
def results():
    answers = session.get('answers', [])
    raw_face = session.get('raw_face', 'neutral')
    
    score = sum(answers)
    face_desc = map_emotion_to_therapy(raw_face)
    
    # Calculate Label
    if score >= 6:
        mood_label = "Mental Well-being: High"
        color = "#2ed573" # Green
    elif score >= 4:
        mood_label = "Mental Well-being: Moderate"
        color = "#ffa502" # Orange
    else:
        mood_label = "Mental Well-being: Needs Care"
        color = "#ff4757" # Red
        
    session['analysis'] = {
        'raw_face': raw_face,
        'face_desc': face_desc,
        'score': score,
        'mood_label': mood_label,
        'color': color
    }
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

# --- TEMPLATES ---

BASE_CSS = """
<style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
    .container { background: white; width: 100%; max-width: 480px; padding: 2rem; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); text-align: center; }
    .btn { background: #0984e3; color: white; border: none; padding: 14px; width: 100%; border-radius: 8px; font-size: 16px; cursor: pointer; margin-top: 15px; font-weight: 600; }
    .btn:disabled { background: #b2bec3; cursor: not-allowed; }
    h1 { color: #2d3436; font-size: 24px; margin-bottom: 10px; }
    .status { color: #636e72; font-size: 14px; margin-bottom: 20px; }
    video { border-radius: 12px; width: 100%; background: #000; transform: scaleX(-1); }
    .option { display: block; padding: 15px; background: #f8f9fa; margin: 10px 0; border-radius: 8px; text-align: left; cursor: pointer; border: 2px solid transparent; transition: 0.2s; }
    .option:hover { border-color: #0984e3; background: #e3f2fd; }
    input[type="radio"] { margin-right: 10px; }
    
    /* Chat Styling */
    .chat-box { height: 400px; overflow-y: auto; padding: 10px; background: #fdfdfd; border: 1px solid #eee; border-radius: 8px; margin-bottom: 15px; display: flex; flex-direction: column; gap: 10px; }
    .msg { padding: 10px 14px; border-radius: 12px; max-width: 80%; font-size: 14px; line-height: 1.5; }
    .msg.You { align-self: flex-end; background: #0984e3; color: white; border-bottom-right-radius: 2px; }
    .msg.Therapist { align-self: flex-start; background: #dfe6e9; color: #2d3436; border-bottom-left-radius: 2px; }
    .input-row { display: flex; gap: 10px; }
    input[type="text"] { flex: 1; padding: 12px; border: 1px solid #ccc; border-radius: 8px; outline: none; }
</style>
"""

HTML_CAMERA = BASE_CSS + """
<div class="container">
    <h1>🧠 Mental Health Check-in</h1>
    <p class="status">Align your face to calibrate the Emotion AI</p>
    
    <div style="position:relative;">
        <video id="video" autoplay playsinline muted></video>
        <canvas id="canvas" style="display:none;"></canvas>
    </div>

    <form id="form" method="POST">
        <input type="hidden" name="image_data" id="image_data">
        <button type="button" class="btn" onclick="capture()" id="snapBtn">Analyze My Vibe</button>
    </form>
</div>
<script>
    const video = document.getElementById('video');
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } })
        .then(s => video.srcObject = s);
        
    function capture() {
        const btn = document.getElementById('snapBtn');
        btn.innerText = "Processing Emotions...";
        btn.disabled = true;
        
        const canvas = document.getElementById('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        
        // JPEG 0.7 is lighter and faster to upload
        document.getElementById('image_data').value = canvas.toDataURL('image/jpeg', 0.7);
        document.getElementById('form').submit();
    }
</script>
"""

HTML_QUIZ = BASE_CSS + """
<div class="container">
    <p style="color:#b2bec3; font-weight:bold;">QUESTION {{ q_num }} / {{ total }}</p>
    <h2>{{ q.text }}</h2>
    <form method="POST">
        {% for txt, val in q.options %}
        <label class="option">
            <input type="radio" name="answer" value="{{ val }}" required> {{ txt }}
        </label>
        {% endfor %}
        <button type="submit" class="btn">Next</button>
    </form>
</div>
"""

HTML_RESULT = BASE_CSS + """
<div class="container">
    <h1>Analysis Complete</h1>
    <div style="margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 12px;">
        <h2 style="color: {{ data.color }}">{{ data.mood_label }}</h2>
        <p><strong>Face AI Detected:</strong> {{ data.face_desc }}</p>
        <p><strong>Quiz Score:</strong> {{ data.score }}/8</p>
    </div>
    <p>Your session is ready.</p>
    <a href="/chat"><button class="btn">Start Therapy</button></a>
</div>
"""

HTML_CHAT = BASE_CSS + """
<div class="container">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <h2>AI Therapist</h2>
        <span style="font-size:12px; background:#eee; padding:4px 8px; border-radius:4px;">{{ data.face_desc }}</span>
    </div>
    <div class="chat-box" id="box">
        <div class="msg Therapist">
            Hello. I've reviewed your results. You seem a bit <strong>{{ data.face_desc }}</strong> today.
            How can I support you right now?
        </div>
        {% for role, txt in history %}
        <div class="msg {{ role }}">{{ txt }}</div>
        {% endfor %}
    </div>
    <form method="POST" class="input-row">
        <input type="text" name="message" placeholder="Type here..." autofocus autocomplete="off" required>
        <button type="submit" class="btn" style="width:auto; margin-top:0;">Send</button>
    </form>
</div>
<script>
    var b = document.getElementById('box');
    b.scrollTop = b.scrollHeight;
</script>
"""

if __name__ == '__main__':
    app.run(port=5000, debug=True)