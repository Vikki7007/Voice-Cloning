from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import torch
import torchaudio
import torchaudio.transforms as T
import uuid
from datetime import datetime
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import logging
import time
import re
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class HindiTextProcessor:
    """Text preprocessing specifically for Hindi"""
    
    def __init__(self):
        # Hindi number words
        self.hindi_numbers = {
            '0': 'शून्य', '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार',
            '5': 'पांच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ'
        }
        
    def normalize_text(self, text):
        """Normalize Hindi text for better TTS output"""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle basic numbers (you can expand this)
        for digit, word in self.hindi_numbers.items():
            text = text.replace(digit, word)
            
        # Remove special characters that might cause issues
        text = re.sub(r'[^\u0900-\u097F\s\.\,\!\?\-]', '', text)
        
        return text

class XTTSHindiTTSAdvanced:
    def __init__(self, model_path="D:/models/TTS/hindiTTS/XTTSv2-Hi_ft/xtts", 
                 config_path="D:/models/TTS/hindiTTS/XTTSv2-Hi_ft/xtts/config.json"):
        """Advanced XTTSv2 Hindi TTS with detailed process control"""
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.text_processor = HindiTextProcessor()
        self.output_dir = "D:/models/TTS/hindiTTS/XTTSv2-Hi_ft/outputs"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_model(self):
        """Load model with detailed logging"""
        print("=" * 60)
        print("LOADING XTTSV2 HINDI MODEL")
        print("=" * 60)
        
        # Check if files exist
        model_file = os.path.join(self.model_path, "model.pth")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        print(f"✓ Model file found: {model_file}")
        print(f"✓ Config file found: {self.config_path}")
        
        # Load configuration
        print("\nLoading configuration...")
        self.config = XttsConfig()
        self.config.load_json(self.config_path)
        
        # Initialize model
        print("Initializing model...")
        self.model = Xtts.init_from_config(self.config)
        
        # Load checkpoint
        print("Loading model weights...")
        self.model.load_checkpoint(self.config, checkpoint_dir=self.model_path, use_deepspeed=False)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            print("Moving model to GPU...")
            self.model.cuda()
            print("✓ Model loaded on GPU")
            print(f"✓ GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠ Model loaded on CPU")
            
        print("✓ Model loading complete!")
        
    def validate_input(self, text: str, language: str, temperature: float, 
                      length_penalty: float, repetition_penalty: float) -> dict:
        """Validate API input parameters"""
        errors = []
        
        # Validate text
        if not text or not text.strip():
            errors.append("Text is required")
        elif len(text) > 250:
            errors.append("Text must be 250 characters or less")
        
        # Validate language
        if language not in ['en', 'hi']:
            errors.append("Language must be 'en' (English) or 'hi' (Hindi)")
        
        # Validate parameters
        if not (0.1 <= temperature <= 1.0):
            errors.append("Temperature must be between 0.1 and 1.0")
        if not (0.5 <= length_penalty <= 2.0):
            errors.append("Length penalty must be between 0.5 and 2.0")
        if not (1.0 <= repetition_penalty <= 10.0):
            errors.append("Repetition penalty must be between 1.0 and 10.0")
        
        return {"valid": len(errors) == 0, "errors": errors}
        
    def ensure_json_serializable(self, obj):
        """Convert any non-JSON serializable objects to serializable types"""
        if isinstance(obj, dict):
            return {key: self.ensure_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self.ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return float(obj.item()) if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)  # Convert everything else to string
        
    def analyze_speaker_audio(self, speaker_wav_path):
        """Analyze and resample speaker reference audio if needed"""
        print(f"\nAnalyzing speaker audio: {speaker_wav_path}")
        
        if not os.path.exists(speaker_wav_path):
            raise FileNotFoundError(f"Speaker audio not found: {speaker_wav_path}")
            
        # Load audio
        waveform, sample_rate = torchaudio.load(speaker_wav_path)
        original_sample_rate = sample_rate
        
        # Resample if sample_rate != 22050
        if sample_rate != 22050:
            print(f"Resampling from {sample_rate} Hz to 22050 Hz...")
            resampler = T.Resample(orig_freq=sample_rate, new_freq=22050)
            waveform = resampler(waveform)
            sample_rate = 22050
            
        duration = waveform.shape[1] / sample_rate
        
        print(f"✓ Duration: {duration:.2f} seconds")
        print(f"✓ Sample rate: {sample_rate} Hz")
        print(f"✓ Channels: {waveform.shape[0]}")
        
        # Quality checks
        warnings = []
        if sample_rate != 22050:
            warnings.append(f"Audio was resampled from {sample_rate} Hz to 22050 Hz")
        if duration < 15:
            warnings.append("Audio is quite short, may affect quality")
        if duration > 60:
            warnings.append("Audio is quite long, may affect performance")
            
        # Return JSON-serializable data + store waveform separately
        return {
            'duration': float(duration),
            'sample_rate': int(sample_rate),
            'channels': int(waveform.shape[0]),
            'warnings': warnings,
            'original_sample_rate': int(original_sample_rate),
            'was_resampled': bool(original_sample_rate != 22050)
        }, waveform  # Return waveform separately
        
    def get_speaker_conditioning(self, speaker_wav_path):
        """Get speaker conditioning with detailed process and resampling support"""
        print(f"\nExtracting speaker characteristics from: {speaker_wav_path}")
        
        # Analyze and potentially resample the audio
        audio_info, waveform = self.analyze_speaker_audio(speaker_wav_path)
        
        # If resampling was done, save the resampled audio temporarily
        if audio_info['was_resampled']:
            temp_resampled_path = speaker_wav_path.replace('.wav', '_resampled_temp.wav')
            if not temp_resampled_path.endswith('_resampled_temp.wav'):
                # Handle other file extensions
                base_name = os.path.splitext(speaker_wav_path)[0]
                temp_resampled_path = f"{base_name}_resampled_temp.wav"
            torchaudio.save(temp_resampled_path, waveform, 22050)
            conditioning_path = temp_resampled_path
        else:
            conditioning_path = speaker_wav_path
        
        print("Extracting speaker embeddings...")
        start_time = time.time()
        
        try:
            # Extract conditioning latents
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=[conditioning_path]
            )
            
            extraction_time = time.time() - start_time
            print(f"✓ Speaker conditioning extracted in {extraction_time:.2f} seconds")
            
            return gpt_cond_latent, speaker_embedding, audio_info
            
        finally:
            # Clean up temporary resampled file if it was created
            if audio_info['was_resampled'] and 'temp_resampled_path' in locals() and os.path.exists(temp_resampled_path):
                try:
                    os.remove(temp_resampled_path)
                except:
                    pass
        
    def synthesize_speech(self, text, language, gpt_cond_latent, speaker_embedding, 
                         temperature=0.7, length_penalty=1.0, repetition_penalty=5.0):
        """Synthesize speech with detailed process"""
        print(f"\nSynthesizing speech in {language}...")
        
        # Preprocess text based on language
        if language == 'hi':
            processed_text = self.text_processor.normalize_text(text)
        else:
            processed_text = text.strip()
        
        print(f"Original text: {text}")
        print(f"Processed text: {processed_text}")
        print(f"Text length: {len(processed_text)} characters")
        
        # Generate speech
        print("Generating speech...")
        start_time = time.time()
        
        out = self.model.inference(
            processed_text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )
        
        synthesis_time = time.time() - start_time
        
        # Process output
        audio_tensor = torch.tensor(out["wav"]).unsqueeze(0)
        audio_duration = len(audio_tensor[0]) / 24000
        rtf = synthesis_time / audio_duration if audio_duration > 0 else 0
        
        print(f"✓ Speech generated in {synthesis_time:.2f} seconds")
        print(f"✓ Audio duration: {audio_duration:.2f} seconds")
        print(f"✓ Real-time factor: {rtf:.2f}x")
        
        # Return JSON-serializable synthesis info
        return audio_tensor, {
            'synthesis_time': float(synthesis_time),
            'audio_duration': float(audio_duration),
            'rtf': float(rtf),
            'temperature': float(temperature),
            'length_penalty': float(length_penalty),
            'repetition_penalty': float(repetition_penalty)
        }
        
    def save_audio(self, audio_tensor, language, text_preview):
        """Save audio with meaningful filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Create meaningful filename
        text_snippet = re.sub(r'[^\w\s-]', '', text_preview)[:30].strip()
        text_snippet = re.sub(r'\s+', '_', text_snippet)
        
        filename = f"voice_clone_{language}_{text_snippet}_{timestamp}_{unique_id}.wav"
        output_path = os.path.join(self.output_dir, filename)
        
        # Save audio
        torchaudio.save(output_path, audio_tensor, 24000)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"✓ Audio saved to: {output_path}")
        print(f"✓ File size: {file_size_mb:.2f} MB")
        
        return output_path, filename, file_size_mb

# Initialize the TTS system
print("Initializing XTTSv2 Hindi TTS System...")
tts_system = XTTSHindiTTSAdvanced()
tts_system.load_model()
print("System ready for voice cloning!")

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'message': 'XTTSv2 Hindi Voice Cloning API is running!',
        'endpoints': {
            'GET /': 'This documentation',
            'GET /health': 'Health check',
            'GET /info': 'API information',
            'POST /synthesize': 'Synthesize voice',
            'GET /download/<filename>': 'Download audio file'
        },
        'usage_example': {
            'method': 'POST',
            'url': '/synthesize',
            'body': {
                'text': 'Your text here (max 250 chars)',
                'language': 'en or hi',
                'temperature': 0.7,
                'length_penalty': 1.0,
                'repetition_penalty': 5.0
            },
            'file': 'audio_file (WAV/MP3/M4A)'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': tts_system.model is not None,
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/info', methods=['GET'])
def get_info():
    """Get API information"""
    return jsonify({
        'api_name': 'XTTSv2 Hindi Voice Cloning API',
        'version': '1.0.0',
        'max_text_length': 250,
        'max_audio_duration': 24,
        'supported_languages': ['en', 'hi'],
        'temperature_range': [0.1, 1.0],
        'length_penalty_range': [0.5, 2.0],
        'repetition_penalty_range': [1.0, 10.0],
        'sample_rate': 24000,
        'model_path': tts_system.model_path,
        'output_dir': tts_system.output_dir
    })

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Main synthesis endpoint that handles file uploads"""
    try:
        # Handle file upload
        if 'audio_file' not in request.files:
            return jsonify({
                'error': 'No audio file uploaded',
                'success': False
            }), 400
            
        audio_file = request.files['audio_file']
        if not audio_file or not audio_file.filename:
            return jsonify({
                'error': 'Invalid audio file',
                'success': False
            }), 400
        
        # Get form data
        text = request.form.get('text', '').strip()
        language = request.form.get('language', 'hi').lower()
        temperature = float(request.form.get('temperature', 0.7))
        length_penalty = float(request.form.get('length_penalty', 1.0))
        repetition_penalty = float(request.form.get('repetition_penalty', 5.0))
        
        # Validate input
        validation = tts_system.validate_input(text, language, temperature, 
                                             length_penalty, repetition_penalty)
        if not validation['valid']:
            return jsonify({
                'error': 'Validation failed',
                'details': validation['errors'],
                'success': False
            }), 400
        
        # Save uploaded file temporarily
        temp_dir = os.path.join(tts_system.output_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_filename = f"temp_{uuid.uuid4().hex[:8]}_{audio_file.filename}"
        temp_filepath = os.path.join(temp_dir, temp_filename)
        audio_file.save(temp_filepath)
        
        try:
            # Get speaker conditioning
            gpt_cond_latent, speaker_embedding, audio_info = tts_system.get_speaker_conditioning(temp_filepath)
            
            # Synthesize speech
            audio_tensor, synthesis_info = tts_system.synthesize_speech(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty
            )
            
            # Save output audio
            output_path, output_filename, file_size_mb = tts_system.save_audio(
                audio_tensor, language, text
            )
            
            # Prepare response - Convert all values to JSON-serializable types
            response_data = {
                'success': True,
                'output_path': str(output_path),
                'output_filename': str(output_filename),
                'synthesis_info': {
                    'duration': float(synthesis_info['audio_duration']),
                    'synthesis_time': float(synthesis_info['synthesis_time']),
                    'rtf': float(synthesis_info['rtf']),
                    'file_size_mb': float(file_size_mb),
                    'language': str(language),
                    'text_length': int(len(text)),
                    'temperature': float(temperature),
                    'length_penalty': float(length_penalty),
                    'repetition_penalty': float(repetition_penalty)
                },
                'audio_info': {
                    'duration': float(audio_info['duration']),
                    'sample_rate': int(audio_info['sample_rate']),
                    'channels': int(audio_info['channels']),
                    'warnings': [str(w) for w in audio_info['warnings']],
                    'original_sample_rate': int(audio_info['original_sample_rate']),
                    'was_resampled': bool(audio_info['was_resampled'])
                },
                'message': 'Voice synthesis completed successfully'
            }
            
            return jsonify(response_data)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:
                    pass
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download generated audio file"""
    try:
        file_path = os.path.join(tts_system.output_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                'error': 'File not found',
                'success': False
            }), 404
        
        return send_file(file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

if __name__ == '__main__':
    try:
        print("🚀 Starting XTTSv2 Hindi Voice Cloning API Server...")
        print("=" * 60)
        print("API Endpoints:")
        print("  GET  / - API documentation")
        print("  GET  /health - Health check")
        print("  GET  /info - API information")
        print("  POST /synthesize - Synthesize voice")
        print("  GET  /download/<filename> - Download audio file")
        print("=" * 60)
        print("✓ Server starting on http://localhost:5000")
        print("✓ Test the API by visiting: http://localhost:5000")
        
        # Run the Flask app
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("Please check if:")
        print("1. Port 5000 is available")
        print("2. All required dependencies are installed")
        print("3. Model files exist at the specified paths")