from fastapi import APIRouter, Depends

from api.dependencies import verify_api_key
from api.runtime import app_state
from api.schemas import ApiActionRequest
from api.services import ActionService

router = APIRouter(dependencies=[Depends(verify_api_key)])
action_service = ActionService(app_state)


@router.post("/api/action")
async def api_action(payload: ApiActionRequest):
    return await action_service.dispatch(payload)
