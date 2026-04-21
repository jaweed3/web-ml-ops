from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.constant import ALLOWED_CONTENT_TYPES, MAX_FILE_SIZE_BYTES
from app.dependencies import check_content_length, require_api_key
from app.pipeline.prediction_pipeline import PredictionPipeline
from app.schema.predict import PredictResponse
from core.logger import get_logger

router = APIRouter(tags=["inference"])
log = get_logger("router.predict")


def _get_pipeline(request: Request) -> PredictionPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    return pipeline


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        401: {"description": "Missing or invalid API key"},
        422: {"description": "Invalid image — wrong MIME type, corrupt bytes, or file too large"},
        503: {"description": "Model not ready yet"},
    },
    dependencies=[Depends(require_api_key), Depends(check_content_length)],
)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictResponse:
    """
    Run victim-detection inference on an uploaded image.

    - Accepts: JPEG, PNG, WebP (max 10 MB)
    - Returns: structured detections with bbox, confidence, and model metadata
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{file.content_type}'. Use jpg, png, or webp.",
        )

    # Read only after MIME check passes — avoids buffering rejected files
    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"File too large ({len(image_bytes) // 1024} KB). Max 10 MB.",
        )

    pipeline = _get_pipeline(request)

    try:
        result = await run_in_threadpool(pipeline.run, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    log.info(
        "prediction_served",
        request_id=result.get("request_id"),
        n_detections=len(result.get("detections", [])),
        filename=file.filename,
    )

    return PredictResponse(
        model_version=result["model_version"],
        model_format=request.app.state.model_format,
        inference_time_ms=result["inference_time_ms"],
        detections=result["detections"],
        image_shape=result["image_shape"],
        request_id=result["request_id"],
    )
