from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' when the process is alive")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")


class ReadyResponse(BaseModel):
    status: str = Field(..., description="'ready' | 'loading' | 'not_initialized'")
    model_version: str | None = Field(None)
    loaded_at: str | None = Field(
        None, description="ISO-8601 UTC timestamp when model became ready"
    )
