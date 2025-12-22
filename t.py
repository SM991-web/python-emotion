import os
import cv2
import numpy as np
import base64
from google import genai
from dotenv import load_dotenv
from flask import Flask, request, render_template_string, redirect, session, url_for

# --- 1. CONFIGURATION & SECRETS ---
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "stable_2025_key")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- 2. INITIALIZE AI CLIENTS ---

# Gemini 2025 Client
client = None
if GEMINI_API_KEY:
    try:
        # Use the NEW Client structure
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini AI Connected (2025 SDK).")
    except Exception as e:
        print(f"⚠️ Gemini Connection Error: {e}")

# DeepFace Vision Setup
VISION_AVAILABLE = False
try:
    from deepface import DeepFace
    print("⏳ Warming up Vision AI...")
    # Prevents category-naming bug by pre-running a blank frame
    blank = np.zeros((224, 224, 3), dtype=np.uint8)
    DeepFace.analyze(blank, actions=['emotion'], enforce_detection=False, silent=True)
    VISION_AVAILABLE = True
    print("✅ Vision AI Ready.")
except Exception as e:
    print(f"⚠️ Vision AI Warning: {e}")

# --- 3. THE 7 QUESTIONS ---
QUESTIONS = {
    1: {"text": "How have you been sleeping lately?", "options": [("Great (7+ hours)", 2), ("Average", 1), ("Poorly", 0)]},
    2: {"text": "How are your energy levels?", "options": [("High", 2), ("Moderate", 1), ("Low", 0)]},
    3: {"text": "Have you felt anxious today?", "options": [("No", 2), ("A little", 1), ("Very much", 0)]},
    4: {"text": "How is your focus?", "options": [("Sharp", 2), ("Distracted", 1), ("Brain fog", 0)]},
    5: {"text": "How is your appetite?", "options": [("Normal", 2), ("Irregular", 1), ("Poor", 0)]},
    6: {"text": "How social have you felt?", "options": [("Social", 2), ("A bit isolated", 1), ("Alone", 0)]},
    7: {"text": "How hopeful are you?", "options": [("Optimistic", 2), ("Neutral", 1), ("Dreading it", 0)]},
}

# --- 4. LOGIC FUNCTIONS ---

def analyze_face(base64_str):
    if not VISION_AVAILABLE or not base64_str: return "neutral"
    try:
        if "," in base64_str: base64_str = base64_str.split(",")[1]
        np_arr = np.frombuffer(base64.b64decode(base64_str), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, silent=True)
        return analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
    except: return "neutral"

def get_ai_reply(user_msg, history, data):
    if not client: return "AI Disconnected."
    
    # SYSTEM CONTEXT (This is the 'Therapist' personality)
    system_instr = (
        f"You are a kind AI therapist. User status: "
        f"Mood is {data['mood_label']} (Quiz Score: {data['score']}/14). "
        f"They look {data['face_desc']} on camera. "
        "Stay empathetic, keep it under 3 sentences, and ask a question."
    )
    
    # Constructing the message contents strictly for the 2025 API
    contents = []
    # Add recent history
    for sender, msg in history[-4:]:
        role = "user" if sender == "You" else "model"
        contents.append({"role": role, "parts": [{"text": msg}]})
    
    # Add the newest message with the system instruction wrapped inside
    contents.append({
        "role": "user", 
        "parts": [{"text": f"SYSTEM_INSTRUCTION: {system_instr}\n\nUSER_SAYS: {user_msg}"}]
    })

    try:
        # ABSOLUTE FIX: Exact model string and generate_content method
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=contents
        )
        return response.text
    except Exception as e:
        print(f"❌ API Error: {e}")
        return "I'm listening, but I had a technical hiccup. Could you try again?"

# --- 5. ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        session.clear()
        session['raw_face'] = analyze_face(request.form.get('image_data'))
        session['answers'] = []
        return redirect(url_for('quiz', q_id=1))
    return render_template_string(HTML_CAMERA)

@app.route('/quiz/<int:q_id>', methods=['GET', 'POST'])
def quiz(q_id):
    if q_id > 7: return redirect(url_for('results'))
    if request.method == 'POST':
        ans = int(request.form['answer'])
        session['answers'] = session.get('answers', []) + [ans]
        return redirect(url_for('quiz', q_id=q_id+1))
    return render_template_string(HTML_QUIZ, q=QUESTIONS[q_id], q_num=q_id)

@app.route('/results')
def results():
    score = sum(session.get('answers', []))
    raw = session.get('raw_face', 'neutral')
    
    # YOUR SCORING LOGIC
    if score <= 7: mood, color = "Distressful", "#ff4757"
    elif 8 <= score <= 11: mood, color = "Needs Guidance", "#ffa502"
    else: mood, color = "Happy & Normal", "#2ed573"

    mapping = {"happy": "Cheerful", "sad": "Somber", "angry": "Frustrated", "fear": "Anxious", "neutral": "Balanced"}
    session['analysis'] = {'face_desc': mapping.get(raw, "Balanced"), 'score': score, 'mood_label': mood, 'color': color}
    session['history'] = []
    return render_template_string(HTML_RESULT, data=session['analysis'])

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    data = session.get('analysis')
    if not data: return redirect(url_for('home'))
    history = session.get('history', [])
    
    if request.method == 'POST':
        user_msg = request.form['message']
        history.append(('You', user_msg))
        reply = get_ai_reply(user_msg, history, data)
        history.append(('Therapist', reply))
        session['history'] = history
        return redirect(url_for('chat'))
    return render_template_string(HTML_CHAT, history=history, data=data)

# --- 6. TEMPLATES ---

CSS = "<style>body{font-family:sans-serif;background:#f4f7f6;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;}.card{background:white;padding:30px;border-radius:20px;width:400px;text-align:center;box-shadow:0 10px 30px rgba(0,0,0,0.05);}.btn{background:#007bff;color:white;border:none;padding:12px;width:100%;border-radius:10px;cursor:pointer;margin-top:15px;font-weight:600;}video{width:100%;border-radius:15px;background:#000;transform:scaleX(-1);}.chat-box{height:300px;overflow-y:auto;border:1px solid #eee;margin:15px 0;padding:10px;display:flex;flex-direction:column;gap:10px;}.msg{padding:10px;border-radius:12px;font-size:14px;max-width:85%;}.You{align-self:flex-end;background:#007bff;color:white;}.Therapist{align-self:flex-start;background:#e9ecef;}</style>"

HTML_CAMERA = CSS + '''<div class="card"><h2>1. Mood Scan</h2><video id="v" autoplay playsinline muted></video><form id="f" method="POST"><input type="hidden" name="image_data" id="i"><button type="button" class="btn" onclick="snap()">Analyze Face</button></form></div><script>const v=document.getElementById("v");navigator.mediaDevices.getUserMedia({video:true}).then(s=>v.srcObject=s).catch(e=>alert("Camera not found!"));function snap(){const c=document.createElement("canvas");c.width=v.videoWidth;c.height=v.videoHeight;c.getContext("2d").drawImage(v,0,0);document.getElementById("i").value=c.toDataURL("image/jpeg",0.6);document.getElementById("f").submit();}</script>'''

HTML_QUIZ = CSS + '<div class="card"><h3>2. Questions ({{q_num}}/7)</h3><p>{{q.text}}</p><form method="POST">{% for txt, val in q.options %}<label style="display:block;margin:10px 0;text-align:left;"><input type="radio" name="answer" value="{{val}}" required> {{txt}}</label>{% endfor %}<button class="btn">Next Question</button></form></div>'

HTML_RESULT = CSS + '<div class="card"><h2>Results Ready</h2><h3 style="color:{{data.color}}">{{data.mood_label}}</h3><p>Detected face: <b>{{data.face_desc}}</b></p><p>Score: {{data.score}}/14</p><a href="/chat"><button class="btn">Start Therapy Chat</button></a></div>'

HTML_CHAT = CSS + '<div class="card"><h3>AI Therapist</h3><div class="chat-box" id="b"><div class="msg Therapist">Hello. Your results show a <b>{{data.mood_label}}</b> mood. I am here for you.</div>{% for s, t in history %}<div class="msg {{s}}">{{t}}</div>{% endfor %}</div><form method="POST" style="display:flex;gap:5px;"><input type="text" name="message" style="flex:1;padding:8px;" placeholder="Type here..." required autofocus><button class="btn" style="width:auto;margin:0;">Send</button></form></div><script>const b=document.getElementById("b");b.scrollTop=b.scrollHeight;</script>'

if __name__ == '__main__':
    app.run(port=5000, debug=True)