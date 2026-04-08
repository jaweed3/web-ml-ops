from pydantic import BaseModel, Field


class BBoxSchema(BaseModel):
    x1: int = Field(..., description="Left edge (pixels)")
    y1: int = Field(..., description="Top edge (pixels)")
    x2: int = Field(..., description="Right edge (pixels)")
    y2: int = Field(..., description="Bottom edge (pixels)")
    cx: int = Field(..., description="Centre x (pixels)")
    cy: int = Field(..., description="Centre y (pixels)")
    width: int = Field(..., description="Box width (pixels)")
    height: int = Field(..., description="Box height (pixels)")


class DetectionSchema(BaseModel):
    class_id: int = Field(..., description="COCO class id")
    class_name: str = Field(..., description="Human-readable class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: BBoxSchema


class PredictResponse(BaseModel):
    model_version: str = Field(..., description="MLflow model version that produced this result")
    model_format: str = Field(..., description="Artifact format: onnx_fp32 | onnx_int8 | tflite_int8")
    inference_time_ms: float = Field(..., description="Pure ONNX inference time (excludes pre/post-processing)")
    detections: list[DetectionSchema] = Field(default_factory=list)
    image_shape: list[int] = Field(..., description="[height, width] of the resized input tensor")
    request_id: str = Field(..., description="Unique request identifier — present in response header X-Request-ID")
