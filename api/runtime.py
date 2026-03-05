from slowapi import Limiter
from slowapi.util import get_remote_address

from api.state import AppState

app_state = AppState()
limiter = Limiter(key_func=get_remote_address)
