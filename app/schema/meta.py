from pydantic import BaseModel, Field


class ModelInfoResponse(BaseModel):
    name: str = Field(..., description="MLflow registered model name")
    version: str = Field(..., description="Resolved model version string")
    format: str = Field(..., description="Artifact format: onnx_fp32 | onnx_int8 | tflite_int8")
    input_shape: list = Field(..., description="ONNX session input shape [N, C, H, W]")
    loaded_at: float | None = Field(None, description="Unix timestamp when session was initialized")


class ModelVersionResponse(BaseModel):
    version: str
