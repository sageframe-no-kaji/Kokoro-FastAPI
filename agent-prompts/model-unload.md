# Task: Model Unload Endpoint

## Branch
`feature/model-unload`

## Goal
Add a `POST /dev/unload` endpoint that releases the Kokoro model from GPU VRAM
without stopping the container. The HTTP server stays alive (zero cold-start
latency on the next request). The model reloads lazily when the next inference
request arrives.

This is needed for homelab deployments where GPU memory must be shared across
services (Ollama, Frigate, etc.) and the TTS container should not hold VRAM
while idle.

## Context
- Upstream issue: https://github.com/remsky/kokoro-fastapi/issues/473
- Fork: https://github.com/sageframe-no-kaji/Kokoro-FastAPI

## Files to change

### `api/src/inference/model_manager.py`
The `ModelManager` singleton holds the model at `self._backend` (a `KokoroV1`
instance). Add an `unload()` async method that:
1. Acquires `self._lock`
2. Calls `await self._backend.unload()` if `_backend` is not None — this handles
   GPU cleanup (torch deallocation)
3. Sets `self._backend = None`
4. Calls `torch.cuda.empty_cache()` after the backend reference is dropped

The lazy-reload path already works: `get_backend()` raises `RuntimeError` if
`_backend is None`, and `generate()` calls `get_backend()` first. You need to
change `generate()` to call `initialize()` lazily if `_backend is None` rather
than raising immediately. Check `initialize()` — it's already idempotent.

```python
async def unload(self) -> None:
    """Release model from GPU memory. Reloads automatically on next request."""
    async with self._lock:
        if self._backend is not None:
            await self._backend.unload()
            self._backend = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded from GPU memory")
```

And in `generate()`, replace the `get_backend()` guard with a lazy initialize:
```python
if not self._backend:
    await self.initialize()
```

### `api/src/routers/development.py`
Add a new endpoint at the end of the file, following the existing pattern:

```python
@router.post("/dev/unload")
async def unload_model(
    tts_service: TTSService = Depends(get_tts_service),
):
    """Release the model from GPU VRAM without stopping the container.

    The model reloads automatically on the next inference request.
    Useful for homelab deployments where GPU memory is shared across services.
    """
    try:
        manager = tts_service._model_manager  # or however TTSService exposes it
        await manager.unload()
        return JSONResponse({"status": "unloaded"})
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
```

**Note:** Verify how `TTSService` exposes the `ModelManager` — may be
`tts_service.model_manager` or accessed via a module-level singleton. Read
`api/src/services/tts_service.py` before writing the endpoint.

## Acceptance criteria
- `POST /dev/unload` returns `{"status": "unloaded"}` with HTTP 200
- After unload, `nvidia-smi` shows no kokoro process holding VRAM
- The next `POST /dev/captioned_speech` request succeeds (model reloads)
- Calling unload when already unloaded is a no-op (no error)
- Concurrent requests during reload don't race (lock prevents double-init)

## Verification commands
```bash
# Start container
docker compose up -d

# Confirm model loaded (watch VRAM)
nvidia-smi

# Unload
curl -X POST http://localhost:8880/dev/unload

# Confirm VRAM released
nvidia-smi

# Confirm reload works
curl -X POST http://localhost:8880/dev/captioned_speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","voice":"bf_emma","input":"Hello world.","stream":false}'

# Confirm VRAM in use again
nvidia-smi
```

## Commit format
```
feat(api): add POST /dev/unload to release model from GPU VRAM

Allows homelab deployments to free GPU memory when the TTS service is idle
without stopping the container. The model reloads lazily on the next request.

Closes #473 (partial)
```
