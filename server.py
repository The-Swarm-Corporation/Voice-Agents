"""
Production-grade FastAPI server for Voice Agents.

This server provides REST API endpoints for:
- Listing available TTS models
- Listing available voices
- Voice agent completions (text-to-speech)
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from voice_agents.main import (
    format_text_for_speech,
    get_media_type_for_format,
    list_models,
    list_voices,
    stream_tts,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Starting Voice Agents API server...")
    logger.info(
        f"Environment: {os.getenv('ENVIRONMENT', 'development')}"
    )

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    yield

    # Shutdown
    logger.info("Shutting down Voice Agents API server...")


# Create FastAPI app
app = FastAPI(
    title="Voice Agents API",
    description="Production-grade API for voice agent text-to-speech services",
    version="0.1.4",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class CompletionRequest(BaseModel):
    """Request model for voice agent completions."""

    text: str = Field(
        ..., description="Text to convert to speech", min_length=1
    )
    model: str = Field(
        default="openai/tts-1",
        description="TTS model identifier (e.g., 'openai/tts-1', 'elevenlabs/eleven_multilingual_v2')",
    )
    voice: Optional[str] = Field(
        default=None,
        description="Voice identifier (e.g., 'alloy', 'rachel'). Defaults based on provider.",
    )
    stream_mode: bool = Field(
        default=False,
        description="If True, process text chunks in real-time as they arrive",
    )
    # OpenAI-specific parameters
    response_format: Optional[str] = Field(
        default=None,
        description="OpenAI audio format: 'pcm', 'mp3', 'opus', 'aac', 'flac'. Default: 'pcm'",
    )
    # ElevenLabs-specific parameters
    voice_id: Optional[str] = Field(
        default=None,
        description="ElevenLabs voice ID (overrides voice parameter for ElevenLabs)",
    )
    stability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="ElevenLabs stability setting (0.0 to 1.0)",
    )
    similarity_boost: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="ElevenLabs similarity boost (0.0 to 1.0)",
    )
    output_format: Optional[str] = Field(
        default=None,
        description="ElevenLabs output format (e.g., 'pcm_44100', 'mp3_44100_128')",
    )
    optimize_streaming_latency: Optional[int] = Field(
        default=None,
        ge=0,
        le=4,
        description="ElevenLabs latency optimization (0-4)",
    )


class ModelResponse(BaseModel):
    """Response model for model listing."""

    models: list[dict[str, str]]


class VoiceResponse(BaseModel):
    """Response model for voice listing."""

    voices: list[dict[str, Optional[str]]]


# Health check endpoint
@app.get("/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "voice-agents-api"}


# List models endpoint
@app.get("/v1/models", response_model=ModelResponse)
async def get_models():
    """
    List all available TTS models with their providers.

    Returns:
        ModelResponse: List of available models with provider information
    """
    try:
        logger.info("Listing available models")
        models = list_models()
        logger.info(f"Found {len(models)} models")
        return ModelResponse(models=models)
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error listing models: {str(e)}"
        )


# List voices endpoint
@app.get("/v1/voices", response_model=VoiceResponse)
async def get_voices():
    """
    List all available TTS voices with their providers.

    Returns:
        VoiceResponse: List of available voices with provider information
    """
    try:
        logger.info("Listing available voices")
        voices = list_voices()
        logger.info(f"Found {len(voices)} voices")
        return VoiceResponse(voices=voices)
    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error listing voices: {str(e)}"
        )


# Voice agent completions endpoint
@app.post("/v1/voice-agent-completions")
async def create_completion(request: CompletionRequest):
    """
    Convert text to speech using the specified model and voice.

    This endpoint streams audio data back to the client. The audio format
    depends on the provider and specified format parameters.

    Args:
        request: CompletionRequest with text, model, voice, and optional parameters

    Returns:
        StreamingResponse: Audio stream with appropriate content type
    """
    try:
        logger.info(
            f"Processing completion request: model={request.model}, "
            f"voice={request.voice}, text_length={len(request.text)}"
        )

        # Format text for speech
        text_chunks = format_text_for_speech(request.text)
        if not text_chunks:
            raise HTTPException(
                status_code=400,
                detail="No valid text chunks after formatting",
            )

        logger.debug(f"Formatted text into {len(text_chunks)} chunks")

        # Determine output format and media type
        output_format = request.output_format
        response_format = request.response_format

        # Determine media type based on model provider
        if request.model.startswith("elevenlabs/"):
            if output_format is None:
                output_format = "pcm_44100"
            media_type = get_media_type_for_format(output_format)
        elif request.model.startswith("openai/"):
            if response_format is None:
                response_format = "pcm"
            # Map OpenAI formats to media types
            format_map = {
                "pcm": "audio/pcm",
                "mp3": "audio/mpeg",
                "opus": "audio/opus",
                "aac": "audio/aac",
                "flac": "audio/flac",
            }
            media_type = format_map.get(response_format, "audio/pcm")
        else:
            # Default to PCM
            media_type = "audio/pcm"

        # Prepare parameters for stream_tts
        stream_params = {
            "text_chunks": text_chunks,
            "model": request.model,
            "voice": request.voice,
            "stream_mode": request.stream_mode,
            "return_generator": True,
        }

        # Add provider-specific parameters
        if request.model.startswith("openai/"):
            if response_format:
                stream_params["response_format"] = response_format
        elif request.model.startswith("elevenlabs/"):
            if voice_id := request.voice_id:
                stream_params["voice_id"] = voice_id
            stream_params["stability"] = request.stability
            stream_params["similarity_boost"] = (
                request.similarity_boost
            )
            if output_format:
                stream_params["output_format"] = output_format
            if request.optimize_streaming_latency is not None:
                stream_params["optimize_streaming_latency"] = (
                    request.optimize_streaming_latency
                )

        # Generate audio stream
        logger.info("Generating audio stream...")
        audio_generator = stream_tts(**stream_params)

        logger.info(f"Streaming audio with media type: {media_type}")

        return StreamingResponse(
            audio_generator,
            media_type=media_type,
            headers={
                "Content-Disposition": "inline; filename=audio",
                "X-Model": request.model,
                "X-Voice": request.voice or "default",
            },
        )

    except ValueError as e:
        logger.error(f"Validation error in completion request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            f"Error processing completion request: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error processing completion: {str(e)}",
        )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Voice Agents API",
        "version": "0.1.4",
        "endpoints": {
            "health": "/v1/health",
            "models": "/v1/models",
            "voices": "/v1/voices",
            "voice-agent-completions": "/v1/voice-agent-completions",
        },
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
