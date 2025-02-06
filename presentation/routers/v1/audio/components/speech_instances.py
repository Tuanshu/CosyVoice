from ..services.audio import AudioService
from ..services.tts_service import TTSService
from ..structures.schemas import OpenAISpeechRequest
from ..core.config import settings
import os
import json
from typing import Dict, List, Union
from loguru import logger


# Load OpenAI mappings
def load_openai_mappings() -> Dict:
    """Load OpenAI voice and model mappings from JSON"""
    api_dir = os.path.dirname(os.path.dirname(__file__))
    mapping_path = os.path.join(api_dir, "core", "openai_mappings.json")
    try:
        with open(mapping_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load OpenAI mappings: {e}")
        return {"models": {}, "voices": {}}


# Global mappings
_openai_mappings = load_openai_mappings()


# Global TTSService instance with lock
_tts_service = None
_init_lock = None


async def get_tts_service() -> TTSService:
    """Get global TTSService instance"""
    global _tts_service, _init_lock

    # Create lock if needed
    if _init_lock is None:
        import asyncio

        _init_lock = asyncio.Lock()

    # Initialize service if needed
    if _tts_service is None:
        async with _init_lock:
            # Double check pattern
            if _tts_service is None:
                _tts_service = await TTSService.create()
                logger.info("Created global TTSService instance")

    return _tts_service


def get_model_name(model: str) -> str:
    """Get internal model name from OpenAI model name"""
    base_name = _openai_mappings["models"].get(model)
    if not base_name:
        raise ValueError(f"Unsupported model: {model}")
    # Add extension based on runtime config
    extension = ".onnx" if settings.use_onnx else ".pth"
    return base_name + extension


async def process_voices(voice_input: Union[str, List[str]], tts_service: TTSService) -> str:
    """Process voice input into a combined voice, handling both string and list formats"""
    # Convert input to list of voices
    if isinstance(voice_input, str):
        # Check if it's an OpenAI voice name
        mapped_voice = _openai_mappings["voices"].get(voice_input)
        if mapped_voice:
            voice_input = mapped_voice
        voices = [v.strip() for v in voice_input.split("+") if v.strip()]
    else:
        # For list input, map each voice if it's an OpenAI voice name
        voices = [_openai_mappings["voices"].get(v, v) for v in voice_input]
        voices = [v.strip() for v in voices if v.strip()]

    if not voices:
        raise ValueError("No voices provided")

    # If single voice, validate and return it
    if len(voices) == 1:
        available_voices = await tts_service.list_voices()
        if voices[0] not in available_voices:
            raise ValueError(f"Voice '{voices[0]}' not found. Available voices: {', '.join(sorted(available_voices))}")
        return voices[0]

    # For multiple voices, validate base voices exist
    available_voices = await tts_service.list_voices()
    for voice in voices:
        if voice not in available_voices:
            raise ValueError(f"Base voice '{voice}' not found. Available voices: {', '.join(sorted(available_voices))}")

    # Combine voices
    return await tts_service.combine_voices(voices=voices)
