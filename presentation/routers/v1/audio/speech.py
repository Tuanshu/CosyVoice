"""OpenAI-compatible router for text-to-speech"""

import json
import os
from typing import AsyncGenerator, Dict, List, Union

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, FileResponse
from loguru import logger

from ..services.audio import AudioService
from ..services.tts_service import TTSService
from ..structures.schemas import OpenAISpeechRequest
from ..core.config import settings

# Load OpenAI mappings
def load_openai_mappings() -> Dict:
    """Load OpenAI voice and model mappings from JSON"""
    api_dir = os.path.dirname(os.path.dirname(__file__))
    mapping_path = os.path.join(api_dir, "core", "openai_mappings.json")
    try:
        with open(mapping_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load OpenAI mappings: {e}")
        return {"models": {}, "voices": {}}

# Global mappings
_openai_mappings = load_openai_mappings()


router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)

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

async def process_voices(
    voice_input: Union[str, List[str]], tts_service: TTSService
) -> str:
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
            raise ValueError(
                f"Voice '{voices[0]}' not found. Available voices: {', '.join(sorted(available_voices))}"
            )
        return voices[0]

    # For multiple voices, validate base voices exist
    available_voices = await tts_service.list_voices()
    for voice in voices:
        if voice not in available_voices:
            raise ValueError(
                f"Base voice '{voice}' not found. Available voices: {', '.join(sorted(available_voices))}"
            )

    # Combine voices
    return await tts_service.combine_voices(voices=voices)


async def stream_audio_chunks(
    tts_service: TTSService, 
    request: OpenAISpeechRequest,
    client_request: Request
) -> AsyncGenerator[bytes, None]:
    """Stream audio chunks as they're generated with client disconnect handling"""
    voice_to_use = await process_voices(request.voice, tts_service)
    
    try:
        async for chunk in tts_service.generate_audio_stream(
            text=request.input,
            voice=voice_to_use,
            speed=request.speed,
            output_format=request.response_format,
        ):
            # Check if client is still connected
            is_disconnected = client_request.is_disconnected
            if callable(is_disconnected):
                is_disconnected = await is_disconnected()
            if is_disconnected:
                logger.info("Client disconnected, stopping audio generation")
                break
            yield chunk
    except Exception as e:
        logger.error(f"Error in audio streaming: {str(e)}")
        # Let the exception propagate to trigger cleanup
        raise


@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
    x_raw_response: str = Header(None, alias="x-raw-response"),
):
    """OpenAI-compatible endpoint for text-to-speech"""
    # Validate model before processing request
    if request.model not in _openai_mappings["models"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_model",
                "message": f"Unsupported model: {request.model}",
                "type": "invalid_request_error"
            }
        )
    
    try:
        # model_name = get_model_name(request.model)
        tts_service = await get_tts_service()
        voice_to_use = await process_voices(request.voice, tts_service)

        # Set content type based on format
        content_type = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }.get(request.response_format, f"audio/{request.response_format}")

        # Check if streaming is requested (default for OpenAI client)
        if request.stream:
            # Create generator but don't start it yet
            generator = stream_audio_chunks(tts_service, request, client_request)
            
            # If download link requested, wrap generator with temp file writer
            if request.return_download_link:
                from ..services.temp_manager import TempFileWriter
                
                temp_writer = TempFileWriter(request.response_format)
                await temp_writer.__aenter__()  # Initialize temp file
                
                # Get download path immediately after temp file creation
                download_path = temp_writer.download_path # 好像temp file不是必要的, 只是暫存; X-Download-Path應該不是必要的header吧
                
                # Create response headers with download path
                headers = {
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Transfer-Encoding": "chunked",
                    "X-Download-Path": download_path
                }

                # Create async generator for streaming
                async def dual_output():
                    try:
                        # Write chunks to temp file and stream
                        async for chunk in generator:
                            if chunk:  # Skip empty chunks
                                await temp_writer.write(chunk)
                                yield chunk

                        # Finalize the temp file
                        await temp_writer.finalize()
                    except Exception as e:
                        logger.error(f"Error in dual output streaming: {e}")
                        await temp_writer.__aexit__(type(e), e, e.__traceback__)
                        raise
                    finally:
                        # Ensure temp writer is closed
                        if not temp_writer._finalized:
                            await temp_writer.__aexit__(None, None, None)

                # Stream with temp file writing
                return StreamingResponse(
                    dual_output(),
                    media_type=content_type,
                    headers=headers
                )
            
            # Standard streaming without download link
            return StreamingResponse(
                generator,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Transfer-Encoding": "chunked"
                }
            )
        else:
            # Generate complete audio using public interface
            audio, _ = await tts_service.generate_audio(
                text=request.input,
                voice=voice_to_use,
                speed=request.speed
            )

            # Convert to requested format with proper finalization
            content = await AudioService.convert_audio(
                audio, 24000, request.response_format,
                is_first_chunk=True,
                is_last_chunk=True
            )

            return Response(
                content=content,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "Cache-Control": "no-cache",  # Prevent caching
                },
            )

    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error"
            }
        )
    except RuntimeError as e:
        # Handle runtime/processing errors
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error"
            }
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in speech generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error"
            }
        )

