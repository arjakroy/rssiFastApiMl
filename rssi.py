from pydantic import BaseModel

class Rssi(BaseModel):
    rssiae: int
    rssi0e: int
    rssi0f: int