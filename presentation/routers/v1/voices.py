from typing import List, Union

from fastapi import HTTPException
from fastapi.responses import FileResponse
from loguru import logger

from ..core.config import settings


@router.get("/download/{filename}")
async def download_audio_file(filename: str):
    """Download a generated audio file from temp storage"""
    try:
        from ..core.paths import _find_file, get_content_type

        # Search for file in temp directory
        file_path = await _find_file(filename=filename, search_paths=[settings.temp_file_dir])

        # Get content type from path helper
        content_type = await get_content_type(file_path)

        return FileResponse(
            file_path,
            media_type=content_type,
            filename=filename,
            headers={"Cache-Control": "no-cache", "Content-Disposition": f"attachment; filename={filename}"},
        )

    except Exception as e:
        logger.error(f"Error serving download file {filename}: {e}")
        raise HTTPException(status_code=500, detail={"error": "server_error", "message": "Failed to serve audio file", "type": "server_error"})


@router.get("/audio/voices")
async def list_voices():
    """List all available voices for text-to-speech"""
    try:
        tts_service = await get_tts_service()
        voices = await tts_service.list_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": "server_error", "message": "Failed to retrieve voice list", "type": "server_error"})


@router.post("/audio/voices/combine")
async def combine_voices(request: Union[str, List[str]]):
    """Combine multiple voices into a new voice.

    Args:
        request: Either a string with voices separated by + (e.g. "voice1+voice2")
                or a list of voice names to combine

    Returns:
        Dict with combined voice name and list of all available voices

    Raises:
        HTTPException:
            - 400: Invalid request (wrong number of voices, voice not found)
            - 500: Server error (file system issues, combination failed)
    """
    try:
        tts_service = await get_tts_service()
        combined_voice = await process_voices(request, tts_service)
        voices = await tts_service.list_voices()
        return {"voices": voices, "voice": combined_voice}

    except ValueError as e:
        logger.warning(f"Invalid voice combination request: {str(e)}")
        raise HTTPException(status_code=400, detail={"error": "validation_error", "message": str(e), "type": "invalid_request_error"})
    except RuntimeError as e:
        logger.error(f"Voice combination processing error: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "processing_error", "message": "Failed to process voice combination request", "type": "server_error"}
        )
    except Exception as e:
        logger.error(f"Unexpected error in voice combination: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": "server_error", "message": "An unexpected error occurred", "type": "server_error"})
