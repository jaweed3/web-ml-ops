from pydantic import BaseModel, Field


class FeedbackBBox(BaseModel):
    x1: int = Field(..., description="Left edge (pixels)")
    y1: int = Field(..., description="Top edge (pixels)")
    x2: int = Field(..., description="Right edge (pixels)")
    y2: int = Field(..., description="Bottom edge (pixels)")


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., description="Request ID from /predict response")
    detections: list[FeedbackBBox] = Field(
        default_factory=list, description="Ground truth bounding boxes"
    )
    annotator: str = Field(default="", description="Optional annotator identifier")


class FeedbackResponse(BaseModel):
    status: str = Field(default="ok")
