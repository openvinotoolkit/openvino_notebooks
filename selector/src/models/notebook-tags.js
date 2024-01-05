export const CATEGORIES = /** @type {const} */ ({
  AI_TRENDS: 'AI Trends',
  FIRST_STEPS: 'First Steps',
  API_OVERVIEW: 'API Overview',
  CONVERT: 'Convert',
  OPTIMIZE: 'Optimize',
  MODEL_DEMOS: 'Model Demos',
  MODEL_TRAINING: 'Model Training',
  LIVE_DEMOS: 'Live Demos',
});

export const TASKS = /** @type {const} */ ({
  MULTIMODAL: {
    TEXT_TO_IMAGE: 'Text-to-Image',
    IMAGE_TO_TEXT: 'Image-to-Text',
    TEXT_TO_VIDEO: 'Text-to-Video',
    VIDEO_TO_TEXT: 'Video-to-Text',
    TEXT_TO_AUDIO: 'Text-to-Audio',
    AUDIO_TO_TEXT: 'Audio-to-Text',
    VISUAL_QUESTION_ANSWERING: 'Visual Question Answering',
  },
  CV: {
    IMAGE_CLASSIFICATION: 'Image Classification',
    IMAGE_SEGMENTATION: 'Image Segmentation',
    IMAGE_GENERATION: 'Image Generation', // Image-to-Image
    OBJECT_DETECTION: 'Object Detection',
    DEPTH_ESTIMTAION: 'Depth Estimation',
    SUPER_RESOLUTION: 'Super Resolution',
    STYLE_TRANSFER: 'Style Transfer',
  },
  NLP: {
    TEXT_CLASSIFICATION: 'Text Classification',
    TEXT_GENERATION: 'Text Generation',
    TRANSLATION: 'Translation',
    SUMMARIZATION: 'Summarization',
    CONVERSATIONAL: 'Conversational',
    ERROR_CORRECTION: 'Error Correction',
    QUESTION_ANSWERING: 'Question Answering',
  },
  AUDIO: {
    TEXT_TO_SPEECH: 'Text-to-Speech',
    SPEECH_TO_TEXT: 'Speech-to-Text',
    SPEECH_RECOGNITION: 'Speech Recognition',
    AUDIO_COMPRESSION: 'Audio Compression',
    VOICE_CONVERSION: 'Voice Conversion',
    AUDIO_GENERATION: 'Audio Generation',
  },
});

export const TASKS_VALUES = Object.values(TASKS)
  .map((v) => Object.values(v))
  .flat();
