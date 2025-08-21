import React, { useState, useRef, useEffect } from 'react';
import { 
  Mic, 
  Upload, 
  Play, 
  Pause, 
  Download, 
  Volume2, 
  Settings,
  Loader2,
  AlertCircle,
  CheckCircle,
  Globe,
  Zap,
  Clock,
  FileAudio
} from 'lucide-react';

// Types
interface SynthesisResult {
  success: boolean;
  output_filename?: string;
  download_url?: string;
  synthesis_info?: {
    duration: number;
    synthesis_time: number;
    rtf: number;
    file_size_mb: number;
    language?: string;
    text_length: number;
    temperature?: number;
    length_penalty?: number;
    repetition_penalty?: number;
    exaggeration?: number;
    cfg_weight?: number;
  };
  audio_info?: {
    duration: number;
    sample_rate: number;
    channels: number;
    warnings: string[];
    original_sample_rate: number;
    was_resampled: boolean;
  };
  reference_audio_info?: {
    duration: number;
    sample_rate: number;
    channels: number;
    warnings: string[];
    file_size_mb: number;
  };
  voice_type?: string;
  parameters?: any;
  message?: string;
  error?: string;
  details?: string[];
}

// Language options
const LANGUAGE_OPTIONS = [
  { 
    code: 'hi', 
    name: 'Hindi', 
    flag: '🇮🇳', 
    description: 'Pure Hindi',
    samples: [
      'नमस्ते दोस्तों, VanniX में आपका स्वागत है!',
      'यह आवाज़ क्लोनिंग तकनीक बहुत अद्भुत है।',
      'भारत में कृत्रिम बुद्धिमत्ता का भविष्य उज्ज्वल है।'
    ]
  },
  { 
    code: 'en', 
    name: 'English', 
    flag: '🇺🇸', 
    description: 'Pure English',
    samples: [
      'Hello everyone, welcome to VanniX voice cloning!',
      'This artificial intelligence technology is amazing.',
      'The future of voice synthesis looks very promising.'
    ]
  },
  { 
    code: 'mixed', 
    name: 'Indo-English', 
    flag: '🌐', 
    description: 'English with Indian accent',
    samples: [
      'Hello friends, welcome to VanniX!',
      'This technology is very amazing!',
      'Voice cloning has a bright future in India.'
    ]
  }
];

function App() {
  // State management
  const [selectedLanguage, setSelectedLanguage] = useState('hi');
  const [text, setText] = useState('');
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<SynthesisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(1);
  
  // XTTSv2 parameters (for Hindi/English)
  const [temperature, setTemperature] = useState(0.7);
  const [lengthPenalty, setLengthPenalty] = useState(1.0);
  const [repetitionPenalty, setRepetitionPenalty] = useState(5.0);
  
  // Chatterbox parameters (for Indo-English)
  const [exaggeration, setExaggeration] = useState(0.5);
  const [cfgWeight, setCfgWeight] = useState(0.5);
  const [apiKey, setApiKey] = useState('cb_test_key_12345');

  // Refs
  const audioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Get current language config
  const currentLanguage = LANGUAGE_OPTIONS.find(lang => lang.code === selectedLanguage) || LANGUAGE_OPTIONS[0];
  
  // Get max text length based on selected API
  const maxTextLength = selectedLanguage === 'mixed' ? 500 : 250;
  
  // Get character count color
  const getCharCountColor = () => {
    const ratio = text.length / maxTextLength;
    if (ratio > 0.9) return 'text-red-500';
    if (ratio > 0.7) return 'text-yellow-500';
    return 'text-gray-500';
  };

  // Handle language change
  const handleLanguageChange = (langCode: string) => {
    setSelectedLanguage(langCode);
    setText(''); // Clear text when switching languages
    setResult(null);
    setError(null);
  };

  // Handle file upload
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Validate file type
      const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/x-m4a'];
      if (!validTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|m4a)$/i)) {
        setError('Please upload a valid audio file (WAV, MP3, or M4A)');
        return;
      }
      
      // Validate file size (max 50MB)
      if (file.size > 50 * 1024 * 1024) {
        setError('File size must be less than 50MB');
        return;
      }
      
      setAudioFile(file);
      setError(null);
    }
  };

  // Handle synthesis
  const handleSynthesize = async () => {
    if (!text.trim()) {
      setError('Please enter some text to synthesize');
      return;
    }

    if (!audioFile) {
      setError('Please upload a reference audio file');
      return;
    }

    if (text.length > maxTextLength) {
      setError(`Text must be ${maxTextLength} characters or less`);
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('text', text);

      // Determine API endpoint and parameters based on language
      let apiUrl: string;
      let headers: HeadersInit = {};

      if (selectedLanguage === 'mixed') {
        // Use Chatterbox API (port 5000) for Indo-English
        apiUrl = 'http://localhost:5000/synthesize';
        headers['X-API-Key'] = apiKey;
        
        formData.append('reference_audio', audioFile);
        formData.append('exaggeration', exaggeration.toString());
        formData.append('cfg_weight', cfgWeight.toString());
      } else {
        // Use XTTSv2 API (port 5001) for Hindi/English
        apiUrl = 'http://localhost:5001/synthesize';
        
        formData.append('audio_file', audioFile);
        formData.append('language', selectedLanguage);
        formData.append('temperature', temperature.toString());
        formData.append('length_penalty', lengthPenalty.toString());
        formData.append('repetition_penalty', repetitionPenalty.toString());
      }

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers,
        body: formData,
      });

      const data: SynthesisResult = await response.json();

      if (data.success) {
        setResult(data);
        
        // Auto-play the generated audio
        if (audioRef.current && data.output_filename) {
          const audioUrl = selectedLanguage === 'mixed' 
            ? `http://localhost:5000/download/${data.output_filename}?api_key=${apiKey}`
            : `http://localhost:5001/download/${data.output_filename}`;
          
          audioRef.current.src = audioUrl;
          audioRef.current.load();
        }
      } else {
        setError(data.error || 'Synthesis failed');
      }
    } catch (err) {
      console.error('Synthesis error:', err);
      setError('Failed to connect to the API. Please make sure the server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle audio playback
  const togglePlayback = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
    }
  };

  // Handle download
  const handleDownload = () => {
    if (result?.output_filename) {
      const downloadUrl = selectedLanguage === 'mixed'
        ? `http://localhost:5000/download/${result.output_filename}?api_key=${apiKey}`
        : `http://localhost:5001/download/${result.output_filename}`;
      
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = result.output_filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  // Handle audio playback with dynamic port routing
  const handlePlayPause = () => {
    if (!audioRef.current || !result?.output_filename) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      const audioUrl = selectedLanguage === 'mixed'
        ? `http://localhost:5000/download/${result.output_filename}?api_key=${apiKey}`
        : `http://localhost:5001/download/${result.output_filename}`;
      
      audioRef.current.src = audioUrl;
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  // Handle sample text selection
  const handleSampleSelect = (sample: string) => {
    setText(sample);
  };

  // Audio event handlers
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
    };
  }, []);

  // Update audio volume
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = volume;
    }
  }, [volume]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl">
              <Mic className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
              VanniX
            </h1>
          </div>
          <p className="text-gray-600 text-lg">
            Advanced Voice Cloning with Multi-Language Support
          </p>
        </div>

        {/* Language Selection */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <Globe className="w-5 h-5" />
            Select Language
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {LANGUAGE_OPTIONS.map((lang) => (
              <button
                key={lang.code}
                onClick={() => handleLanguageChange(lang.code)}
                className={`p-4 rounded-xl border-2 transition-all duration-200 ${
                  selectedLanguage === lang.code
                    ? 'border-indigo-500 bg-indigo-50 shadow-md'
                    : 'border-gray-200 hover:border-indigo-300 hover:bg-gray-50'
                }`}
              >
                <div className="text-3xl mb-2">{lang.flag}</div>
                <div className="font-semibold text-gray-800">{lang.name}</div>
                <div className="text-sm text-gray-600">{lang.description}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Sample Texts */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">
            Sample Texts for {currentLanguage.name}
          </h3>
          <div className="space-y-2">
            {currentLanguage.samples.map((sample, index) => (
              <button
                key={index}
                onClick={() => handleSampleSelect(sample)}
                className="w-full text-left p-3 rounded-lg border border-gray-200 hover:border-indigo-300 hover:bg-indigo-50 transition-colors duration-200"
              >
                <span className="text-gray-700">{sample}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Text Input */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-800">Enter Text</h2>
            <span className={`text-sm font-medium ${getCharCountColor()}`}>
              {text.length}/{maxTextLength}
            </span>
          </div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={`Enter text in ${currentLanguage.name} (max ${maxTextLength} characters)...`}
            className="w-full h-32 p-4 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none"
            maxLength={maxTextLength}
          />
        </div>

        {/* Audio Upload */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Upload Reference Audio
          </h2>
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-indigo-400 hover:bg-indigo-50 transition-colors duration-200 cursor-pointer"
          >
            <FileAudio className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            {audioFile ? (
              <div>
                <p className="text-green-600 font-medium">{audioFile.name}</p>
                <p className="text-sm text-gray-500">
                  {(audioFile.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
            ) : (
              <div>
                <p className="text-gray-600 mb-2">Click to upload reference audio</p>
                <p className="text-sm text-gray-500">Supports WAV, MP3, M4A (max 50MB)</p>
              </div>
            )}
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.m4a,audio/*"
            onChange={handleFileUpload}
            className="hidden"
          />
        </div>

        {/* API-Specific Parameters */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5" />
            {selectedLanguage === 'mixed' ? 'Chatterbox Settings' : 'XTTSv2 Settings'}
          </h2>
          
          {selectedLanguage === 'mixed' ? (
            // Chatterbox API parameters
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  API Key
                </label>
                <input
                  type="text"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  placeholder="Enter your Chatterbox API key"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Exaggeration: {exaggeration.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1.5"
                  step="0.1"
                  value={exaggeration}
                  onChange={(e) => setExaggeration(parseFloat(e.target.value))}
                  className="slider w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Natural (0.0)</span>
                  <span>Expressive (1.5)</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  CFG Weight: {cfgWeight.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={cfgWeight}
                  onChange={(e) => setCfgWeight(parseFloat(e.target.value))}
                  className="slider w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Creative (0.1)</span>
                  <span>Precise (1.0)</span>
                </div>
              </div>
            </div>
          ) : (
            // XTTSv2 API parameters
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Temperature: {temperature.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="slider w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Stable (0.1)</span>
                  <span>Creative (1.0)</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Length Penalty: {lengthPenalty.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  value={lengthPenalty}
                  onChange={(e) => setLengthPenalty(parseFloat(e.target.value))}
                  className="slider w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Shorter (0.5)</span>
                  <span>Longer (2.0)</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Repetition Penalty: {repetitionPenalty.toFixed(1)}
                </label>
                <input
                  type="range"
                  min="1.0"
                  max="10.0"
                  step="0.5"
                  value={repetitionPenalty}
                  onChange={(e) => setRepetitionPenalty(parseFloat(e.target.value))}
                  className="slider w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Allow Repetition (1.0)</span>
                  <span>Avoid Repetition (10.0)</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Generate Button */}
        <div className="mb-6">
          <button
            onClick={handleSynthesize}
            disabled={isLoading || !text.trim() || !audioFile}
            className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-4 px-6 rounded-xl font-semibold text-lg hover:from-indigo-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center gap-3"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-6 h-6 animate-spin" />
                Generating Speech...
              </>
            ) : (
              <>
                <Zap className="w-6 h-6" />
                Generate Speech
              </>
            )}
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-6 animate-in slide-in-from-top duration-300">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
              <p className="text-red-700">{error}</p>
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="bg-white rounded-2xl shadow-lg p-6 animate-in slide-in-from-right duration-500">
            <div className="flex items-center gap-3 mb-4">
              <CheckCircle className="w-6 h-6 text-green-500" />
              <h2 className="text-xl font-semibold text-gray-800">Generated Audio</h2>
            </div>

            {/* Audio Player */}
            <div className="bg-gray-50 rounded-xl p-4 mb-4">
              <div className="flex items-center gap-4 mb-3">
                <button
                  onClick={handlePlayPause}
                  className="p-3 bg-indigo-600 text-white rounded-full hover:bg-indigo-700 transition-colors duration-200"
                >
                  {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6" />}
                </button>
                
                <div className="flex items-center gap-2 flex-1">
                  <Volume2 className="w-5 h-5 text-gray-600" />
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={volume}
                    onChange={(e) => setVolume(parseFloat(e.target.value))}
                    className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                <button
                  onClick={handleDownload}
                  className="p-3 bg-green-600 text-white rounded-full hover:bg-green-700 transition-colors duration-200"
                >
                  <Download className="w-6 h-6" />
                </button>
              </div>
              
              <audio ref={audioRef} className="hidden" />
            </div>

            {/* Synthesis Info */}
            {result.synthesis_info && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div className="bg-blue-50 p-3 rounded-lg">
                  <div className="flex items-center gap-2 text-blue-600 mb-1">
                    <Clock className="w-4 h-4" />
                    <span className="font-medium">Duration</span>
                  </div>
                  <span className="text-blue-800 font-semibold">
                    {(result.synthesis_info.duration || result.synthesis_info.audio_duration || 0).toFixed(2)}s
                  </span>
                </div>
                
                <div className="bg-green-50 p-3 rounded-lg">
                  <div className="flex items-center gap-2 text-green-600 mb-1">
                    <Zap className="w-4 h-4" />
                    <span className="font-medium">Generation</span>
                  </div>
                  <span className="text-green-800 font-semibold">
                    {result.synthesis_info.synthesis_time.toFixed(2)}s
                  </span>
                </div>
                
                <div className="bg-purple-50 p-3 rounded-lg">
                  <div className="flex items-center gap-2 text-purple-600 mb-1">
                    <FileAudio className="w-4 h-4" />
                    <span className="font-medium">File Size</span>
                  </div>
                  <span className="text-purple-800 font-semibold">
                    {result.synthesis_info.file_size_mb?.toFixed(2) || 'N/A'} MB
                  </span>
                </div>
                
                <div className="bg-orange-50 p-3 rounded-lg">
                  <div className="flex items-center gap-2 text-orange-600 mb-1">
                    <Settings className="w-4 h-4" />
                    <span className="font-medium">RTF</span>
                  </div>
                  <span className="text-orange-800 font-semibold">
                    {result.synthesis_info.rtf.toFixed(2)}x
                  </span>
                </div>
              </div>
            )}

            {/* Audio Warnings */}
            {result.audio_info?.warnings && result.audio_info.warnings.length > 0 && (
              <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                <h4 className="font-medium text-yellow-800 mb-2">Audio Analysis:</h4>
                <ul className="text-sm text-yellow-700 space-y-1">
                  {result.audio_info.warnings.map((warning, index) => (
                    <li key={index}>• {warning}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;