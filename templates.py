HTML_CAMERA = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>Camera Upload</title></head>
<body>
  <h1>Upload or Take a Photo</h1>
  <form method="post">
    <input type="file" id="file" accept="image/*" onchange="encodeImageFileAsURL(this)" />
    <input type="hidden" name="image_data" id="image_data" />
    <button type="submit">Submit</button>
  </form>
  <p>Choose a file to upload; it will be sent as a base64 data URL.</p>
  <script>
    function encodeImageFileAsURL(element){
      var file = element.files[0];
      var reader = new FileReader();
      reader.onloadend = function(){ document.getElementById('image_data').value = reader.result; };
      if(file) reader.readAsDataURL(file);
    }
  </script>
</body>
</html>
"""

HTML_QUIZ = """<!doctype html>
<html>
<body>
  <h2>Question {{ q_num }} / {{ total }}</h2>
  <form method="post">
    <p>{{ q['text'] }}</p>
    {% for label, val in q['options'] %}
      <div>
        <label><input type="radio" name="answer" value="{{ val }}" required> {{ label }}</label>
      </div>
    {% endfor %}
    <button type="submit">Next</button>
  </form>
</body>
</html>
"""

HTML_RESULT = """<!doctype html>
<html>
<body>
  <h1>Results</h1>
  <p><strong>Mood:</strong> {{ data['face_desc'] }}</p>
  <p><strong>Score:</strong> {{ data['score'] }} / 14</p>
  <p><strong>Status:</strong> {{ data['mood_label'] }}</p>
  <a href="/chat">Talk to the therapist bot</a>
</body>
</html>
"""

HTML_CHAT = """<!doctype html>
<html>
<body>
  <h1>Therapist Chat</h1>
  <div id="history">
    {% for sender, msg in history %}
      <p><strong>{{ sender }}:</strong> {{ msg }}</p>
    {% endfor %}
  </div>
  <form method="post">
    <input name="message" placeholder="Say something..." required />
    <button type="submit">Send</button>
  </form>
</body>
</html>
"""
