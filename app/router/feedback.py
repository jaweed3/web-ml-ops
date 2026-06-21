from fastapi import APIRouter, Depends

from app.components.prediction_logger import log_feedback
from app.dependencies import require_api_key
from app.schema.feedback import FeedbackRequest, FeedbackResponse

router = APIRouter(tags=["feedback"])


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    dependencies=[Depends(require_api_key)],
)
def feedback(body: FeedbackRequest) -> FeedbackResponse:
    """Submit ground-truth labels for a previous prediction request."""
    log_feedback(
        request_id=body.request_id,
        ground_truth=[d.model_dump() for d in body.detections],
        annotator=body.annotator,
    )
    return FeedbackResponse(status="ok")
