from fastapi import APIRouter
from .speech import router as speech_router

router = APIRouter()
router.include_router(speech_router, prefix="/speech")
