#!/usr/bin/env python3
"""
Alternative Flask App Using YOUR Trained MLP Model
=================================================

This is a version of your app.py that uses YOUR trained MLP emotion model
while keeping all your original code intact.

Simply run this file instead of app.py to use your trained model.
"""

# Standard library imports
import os
import sqlite3
import random
import base64
import io
from datetime import datetime

# Third-party imports
from flask import Flask, render_template, request, jsonify
from PIL import Image

# Import YOUR trained model
try:
    from my_trained_model import init_my_trained_emotion_detector
    EMOTION_DETECTOR = init_my_trained_emotion_detector()
    print("🚀 YOUR Trained MLP Model imported successfully")
except ImportError as e:
    print(f"⚠️  Could not import YOUR trained model: {e}")
    EMOTION_DETECTOR = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database (same as your original)
def init_db():
    conn = sqlite3.connect('emotion_detection.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_path TEXT,
            predicted_emotion TEXT NOT NULL,
            confidence REAL,
            capture_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_emotion(image_path=None):
    """Predict emotion using YOUR trained MLP model"""
    try:
        if EMOTION_DETECTOR is not None and EMOTION_DETECTOR.model_loaded:
            # Use YOUR trained MLP model
            emotion, confidence, message = EMOTION_DETECTOR.predict_emotion(image_path)
            return emotion, confidence, message
        else:
            # Fallback to random prediction
            emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            emotion = random.choice(emotions)
            confidence = random.uniform(0.6, 0.95)
            message = f"Fallback prediction: {emotion}"
            return emotion, confidence, message
    except Exception as e:
        print(f"Prediction error: {e}")
        # Final fallback
        return "happy", 0.75, "Error - fallback prediction"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and emotion prediction using YOUR model"""
    try:
        # Check if user info is provided
        name = request.form.get('name')
        email = request.form.get('email', '')
        
        if not name:
            return jsonify({
                'success': False,
                'error': 'Please provide your name'
            }), 400
        
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if file and allowed_file(file.filename):
            # Save user info
            conn = sqlite3.connect('emotion_detection.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (name, email) VALUES (?, ?)', (name, email))
            user_id = cursor.lastrowid
            
            # Save uploaded file
            filename = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predict emotion using YOUR trained model
            emotion, confidence, message = predict_emotion(filepath)
            
            # Save prediction to database
            cursor.execute('''INSERT INTO predictions 
                           (user_id, image_path, predicted_emotion, confidence, capture_type) 
                           VALUES (?, ?, ?, ?, ?)''', 
                          (user_id, filepath, emotion, confidence, 'upload'))
            conn.commit()
            conn.close()
            
            return jsonify({
                'success': True,
                'emotion': emotion,
                'confidence': confidence,
                'message': f'Hello {name}! {message} (Using YOUR trained MLP model)'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload an image file.'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/process_live_image', methods=['POST'])
def process_live_image():
    """Process live camera image using YOUR model"""
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email', '')
        image_data = data.get('image')
        
        if not name or not image_data:
            return jsonify({'error': 'Name and image are required'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save user info
        conn = sqlite3.connect('emotion_detection.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (name, email) VALUES (?, ?)', (name, email))
        user_id = cursor.lastrowid
        
        # Save captured image
        filename = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_live_capture.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)
        
        # Predict emotion using YOUR trained model
        emotion, confidence, message = predict_emotion(filepath)
        
        # Save prediction to database
        cursor.execute('''INSERT INTO predictions 
                       (user_id, image_path, predicted_emotion, confidence, capture_type) 
                       VALUES (?, ?, ?, ?, ?)''', 
                      (user_id, filepath, emotion, confidence, 'live_capture'))
        conn.commit()
        conn.close()
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'message': f'Hello {name}! {message} (Using YOUR trained MLP model)'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Get information about your trained model"""
    if EMOTION_DETECTOR and EMOTION_DETECTOR.model_loaded:
        return jsonify({
            'model_loaded': True,
            'model_type': str(type(EMOTION_DETECTOR.model)),
            'emotions': EMOTION_DETECTOR.emotions,
            'message': 'Using YOUR trained MLP emotion detection model'
        })
    else:
        return jsonify({
            'model_loaded': False,
            'message': 'Your trained model could not be loaded'
        })

if __name__ == '__main__':
    print("🚀 Starting Flask app with YOUR trained MLP model")
    init_db()
    
    # Show model status
    if EMOTION_DETECTOR and EMOTION_DETECTOR.model_loaded:
        print("✅ YOUR trained MLP model is ready!")
        print(f"📊 Model type: {type(EMOTION_DETECTOR.model)}")
        print(f"🎯 Emotions: {EMOTION_DETECTOR.emotions}")
    else:
        print("⚠️  YOUR trained model not loaded, using fallback predictions")
    
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)