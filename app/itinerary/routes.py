from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class RouteGeometryRequest(BaseModel):
    waypoints: list[list[float]]
    mode: str = "car"


class RouteGeometryResponse(BaseModel):
    points: list[list[float]]
    duration_s: int = 0
    distance_m: float = 0.0


@router.post("/geometry", response_model=RouteGeometryResponse)
async def route_geometry(body: RouteGeometryRequest):
    from app.tools.trackasia import get_route_geometry

    pairs = [(w[0], w[1]) for w in body.waypoints if len(w) >= 2]
    if len(pairs) < 2:
        return RouteGeometryResponse(points=[])
    result = get_route_geometry(pairs, travel_mode=body.mode)
    if not result:
        return RouteGeometryResponse(points=[])
    return RouteGeometryResponse(
        points=result["points"],
        duration_s=result.get("duration_s") or 0,
        distance_m=result.get("distance_m") or 0.0,
    )
