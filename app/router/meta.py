from fastapi import APIRouter, Depends, Request

from app.components.runner import ONNXRunner
from app.dependencies import get_runner, require_api_key
from app.schema.meta import ModelInfoResponse, ModelVersionResponse

router = APIRouter(tags=["model"], dependencies=[Depends(require_api_key)])


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(request: Request, runner: ONNXRunner = Depends(get_runner)) -> ModelInfoResponse:
    return ModelInfoResponse(
        name=request.app.state.model_name,
        version=runner.version,
        format=request.app.state.model_format,
        input_shape=runner.input_shape,
        loaded_at=runner.loaded_at,
    )


@router.get("/model/version", response_model=ModelVersionResponse)
def model_version(runner: ONNXRunner = Depends(get_runner)) -> ModelVersionResponse:
    return ModelVersionResponse(version=runner.version)
