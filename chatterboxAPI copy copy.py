from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import torch
import torchaudio
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
import time
import re
import json
import logging
from chatterbox.tts import ChatterboxTTS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# API Keys storage (in production, use a proper database)
API_KEYS = {
    "cb_test_key_12345": {
        "name": "Test Key",
        "created": datetime.now().isoformat(),
        "requests_made": 0,
        "daily_limit": 100,
        "last_reset": datetime.now().date().isoformat()
    }
}

# Generate a new API key
def generate_api_key():
    """Generate a new API key"""
    return f"cb_{secrets.token_hex(16)}"

# API Key authentication decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return jsonify({
                'error': 'API key required',
                'message': 'Include API key in X-API-Key header or api_key parameter'
            }), 401
        
        if api_key not in API_KEYS:
            return jsonify({
                'error': 'Invalid API key',
                'message': 'Please check your API key'
            }), 401
        
        # Check daily limits
        key_info = API_KEYS[api_key]
        today = datetime.now().date().isoformat()
        
        if key_info['last_reset'] != today:
            key_info['requests_made'] = 0
            key_info['last_reset'] = today
        
        if key_info['requests_made'] >= key_info['daily_limit']:
            return jsonify({
                'error': 'Daily limit exceeded',
                'message': f'Daily limit of {key_info["daily_limit"]} requests reached'
            }), 429
        
        # Increment request count
        key_info['requests_made'] += 1
        
        return f(*args, **kwargs)
    
    return decorated_function

class ChatterboxTTSAPI:
    def __init__(self):
        """Initialize Chatterbox TTS API"""
        self.model = None
        self.output_dir = "D:/models/indi-english/chatterbox/api_outputs"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_model(self):
        """Load Chatterbox TTS model"""
        print("=" * 60)
        print("LOADING CHATTERBOX TTS MODEL")
        print("=" * 60)
        
        print(f"Device: {self.device}")
        
        try:
            print("Loading Chatterbox TTS model...")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            print("✅ Chatterbox TTS model loaded successfully!")
            
            if self.device == "cuda":
                print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
            else:
                print("⚠️  Using CPU (consider using GPU for better performance)")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def validate_input(self, text: str, exaggeration: float, cfg_weight: float) -> dict:
        """Validate API input parameters"""
        errors = []
        
        # Validate text
        if not text or not text.strip():
            errors.append("Text is required")
        elif len(text) > 500:
            errors.append("Text must be 500 characters or less")
        
        # Validate parameters
        if not (0.0 <= exaggeration <= 1.5):
            errors.append("Exaggeration must be between 0.0 and 1.5")
        if not (0.1 <= cfg_weight <= 1.0):
            errors.append("CFG weight must be between 0.1 and 1.0")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def analyze_reference_audio(self, audio_path):
        """Analyze reference audio file"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")
        
        try:
            # Load audio to get info
            waveform, sample_rate = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sample_rate
            
            info = {
                'duration': float(duration),
                'sample_rate': int(sample_rate),
                'channels': int(waveform.shape[0]),
                'file_size_mb': float(os.path.getsize(audio_path) / (1024 * 1024))
            }
            
            # Quality warnings
            warnings = []
            if duration < 3:
                warnings.append("Audio shorter than 3 seconds may affect quality")
            if duration > 30:
                warnings.append("Audio longer than 30 seconds may slow processing")
            if sample_rate != 24000:
                warnings.append(f"Audio sample rate is {sample_rate}Hz, model expects 24kHz")
            
            info['warnings'] = warnings
            
            return info
            
        except Exception as e:
            raise Exception(f"Error analyzing audio: {e}")
    
    def synthesize_speech(self, text, reference_audio_path=None, exaggeration=0.5, cfg_weight=0.5):
        """Synthesize speech using Chatterbox TTS"""
        print(f"\n🎤 Synthesizing speech...")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"Reference audio: {reference_audio_path if reference_audio_path else 'None (default voice)'}")
        print(f"Exaggeration: {exaggeration}, CFG Weight: {cfg_weight}")
        
        start_time = time.time()
        
        try:
            # Generate speech
            if reference_audio_path:
                wav = self.model.generate(
                    text,
                    audio_prompt_path=reference_audio_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight
                )
            else:
                wav = self.model.generate(
                    text,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight
                )
            
            synthesis_time = time.time() - start_time
            audio_duration = len(wav[0]) / self.model.sr
            rtf = synthesis_time / audio_duration if audio_duration > 0 else 0
            
            print(f"✅ Synthesis completed in {synthesis_time:.2f}s")
            print(f"✅ Audio duration: {audio_duration:.2f}s")
            print(f"✅ Real-time factor: {rtf:.2f}x")
            
            return wav, {
                'synthesis_time': float(synthesis_time),
                'audio_duration': float(audio_duration),
                'rtf': float(rtf),
                'sample_rate': int(self.model.sr)
            }
            
        except Exception as e:
            print(f"❌ Synthesis error: {e}")
            raise
    
    def save_audio(self, audio_tensor, text_preview, voice_type="default"):
        """Save audio with meaningful filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Create meaningful filename
        text_snippet = re.sub(r'[^\w\s-]', '', text_preview)[:30].strip()
        text_snippet = re.sub(r'\s+', '_', text_snippet)
        
        filename = f"chatterbox_{voice_type}_{text_snippet}_{timestamp}_{unique_id}.wav"
        output_path = os.path.join(self.output_dir, filename)
        
        # Save audio
        torchaudio.save(output_path, audio_tensor, self.model.sr)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"✅ Audio saved: {filename}")
        print(f"✅ File size: {file_size_mb:.2f} MB")
        
        return output_path, filename, file_size_mb

# Initialize TTS system
print("Initializing Chatterbox TTS API...")
tts_system = ChatterboxTTSAPI()
tts_system.load_model()
print("🚀 Chatterbox TTS API ready!")

# API Routes
@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        'name': 'Chatterbox TTS API',
        'version': '1.0.0',
        'description': 'Text-to-Speech API with voice cloning using Chatterbox TTS',
        'endpoints': {
            'GET /': 'This documentation',
            'GET /health': 'Health check',
            'GET /info': 'API information',
            'POST /api-key': 'Generate new API key',
            'POST /synthesize': 'Synthesize speech',
            'GET /download/<filename>': 'Download audio file'
        },
        'authentication': 'Include API key in X-API-Key header or api_key parameter',
        'usage_example': {
            'url': '/synthesize',
            'method': 'POST',
            'headers': {'X-API-Key': 'your_api_key_here'},
            'body': {
                'text': 'Hello world!',
                'exaggeration': 0.5,
                'cfg_weight': 0.5
            },
            'file_upload': 'reference_audio (optional WAV file for voice cloning)'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': tts_system.model is not None,
        'device': tts_system.device,
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/info', methods=['GET'])
def get_info():
    """API information"""
    return jsonify({
        'api_name': 'Chatterbox TTS API',
        'model': 'Chatterbox TTS by Resemble AI',
        'version': '1.0.0',
        'max_text_length': 500,
        'supported_audio_formats': ['WAV', 'MP3', 'M4A'],
        'exaggeration_range': [0.0, 1.5],
        'cfg_weight_range': [0.1, 1.0],
        'sample_rate': 24000,
        'features': [
            'Text-to-speech synthesis',
            'Voice cloning with reference audio',
            'Emotion/exaggeration control',
            'High-quality neural speech synthesis',
            'GPU acceleration support'
        ]
    })

@app.route('/api-key', methods=['POST'])
def create_api_key():
    """Generate new API key"""
    data = request.get_json() or {}
    key_name = data.get('name', 'Unnamed Key')
    daily_limit = data.get('daily_limit', 100)
    
    # Generate new key
    new_key = generate_api_key()
    
    # Store key info
    API_KEYS[new_key] = {
        'name': key_name,
        'created': datetime.now().isoformat(),
        'requests_made': 0,
        'daily_limit': daily_limit,
        'last_reset': datetime.now().date().isoformat()
    }
    
    return jsonify({
        'success': True,
        'api_key': new_key,
        'name': key_name,
        'daily_limit': daily_limit,
        'message': 'API key generated successfully. Keep it secure!'
    })

@app.route('/synthesize', methods=['POST'])
@require_api_key
def synthesize():
    """Main TTS synthesis endpoint"""
    try:
        # Handle form data and file upload
        text = request.form.get('text', '').strip()
        exaggeration = float(request.form.get('exaggeration', 0.5))
        cfg_weight = float(request.form.get('cfg_weight', 0.5))
        
        # Validate input
        validation = tts_system.validate_input(text, exaggeration, cfg_weight)
        if not validation['valid']:
            return jsonify({
                'success': False,
                'error': 'Validation failed',
                'details': validation['errors']
            }), 400
        
        # Handle optional reference audio
        reference_audio_path = None
        reference_audio_info = None
        
        if 'reference_audio' in request.files:
            audio_file = request.files['reference_audio']
            if audio_file and audio_file.filename:
                # Save uploaded file temporarily
                temp_filename = f"temp_ref_{uuid.uuid4().hex[:8]}_{audio_file.filename}"
                temp_filepath = os.path.join(tts_system.output_dir, temp_filename)
                audio_file.save(temp_filepath)
                
                try:
                    # Analyze reference audio
                    reference_audio_info = tts_system.analyze_reference_audio(temp_filepath)
                    reference_audio_path = temp_filepath
                    
                except Exception as e:
                    # Clean up and return error
                    if os.path.exists(temp_filepath):
                        os.remove(temp_filepath)
                    return jsonify({
                        'success': False,
                        'error': f'Reference audio error: {str(e)}'
                    }), 400
        
        try:
            # Synthesize speech
            audio_tensor, synthesis_info = tts_system.synthesize_speech(
                text=text,
                reference_audio_path=reference_audio_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
            
            # Save output
            voice_type = "cloned" if reference_audio_path else "default"
            output_path, filename, file_size_mb = tts_system.save_audio(
                audio_tensor, text, voice_type
            )
            
            # Prepare response
            response = {
                'success': True,
                'output_filename': filename,
                'download_url': f'/download/{filename}',
                'synthesis_info': synthesis_info,
                'file_size_mb': file_size_mb,
                'voice_type': voice_type,
                'parameters': {
                    'text_length': len(text),
                    'exaggeration': exaggeration,
                    'cfg_weight': cfg_weight
                }
            }
            
            if reference_audio_info:
                response['reference_audio_info'] = reference_audio_info
            
            return jsonify(response)
            
        finally:
            # Clean up temporary reference audio
            if reference_audio_path and os.path.exists(reference_audio_path):
                try:
                    os.remove(reference_audio_path)
                except:
                    pass
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/download/<filename>', methods=['GET'])
@require_api_key
def download_file(filename):
    """Download generated audio file"""
    try:
        file_path = os.path.join(tts_system.output_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        return send_file(file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/usage', methods=['GET'])
@require_api_key
def get_usage():
    """Get API usage statistics"""
    api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
    key_info = API_KEYS[api_key]
    
    return jsonify({
        'api_key_name': key_info['name'],
        'requests_made_today': key_info['requests_made'],
        'daily_limit': key_info['daily_limit'],
        'requests_remaining': key_info['daily_limit'] - key_info['requests_made'],
        'last_reset': key_info['last_reset'],
        'created': key_info['created']
    })

if __name__ == '__main__':
    try:
        print("🚀 Starting Chatterbox TTS API Server...")
        print("=" * 60)
        print("Available Endpoints:")
        print("  GET  / - API documentation")
        print("  GET  /health - Health check")
        print("  GET  /info - API information")
        print("  POST /api-key - Generate API key")
        print("  POST /synthesize - Synthesize speech (requires API key)")
        print("  GET  /download/<filename> - Download audio (requires API key)")
        print("  GET  /usage - Check API usage (requires API key)")
        print("=" * 60)
        print("✅ Test API key: cb_test_key_12345")
        print("✅ Server starting on http://localhost:5000")
        print("✅ Visit http://localhost:5000 for documentation")
        
        app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        logger.error(f"Server startup error: {e}")