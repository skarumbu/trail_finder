import asyncio
import json
import math
import os
import logging
import time
import uuid
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

import httpx
from openai import AzureOpenAI
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("trail-finder")

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        rid = str(uuid.uuid4())[:8]
        try:
            response = await call_next(request)
            logger.info(json.dumps({
                "event": "request",
                "service": "trail-finder",
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.time() - start) * 1000, 1),
                "request_id": rid,
            }))
            return response
        except Exception as exc:
            logger.error(json.dumps({
                "event": "error",
                "service": "trail-finder",
                "method": request.method,
                "path": request.url.path,
                "error": str(exc),
                "duration_ms": round((time.time() - start) * 1000, 1),
                "request_id": rid,
            }))
            raise

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.add_middleware(MetricsMiddleware)

GOOGLE_PLACES_API_KEY = os.environ["GOOGLE_PLACES_API_KEY"]
GOOGLE_SEARCH_API_KEY = os.environ["GOOGLE_SEARCH_API_KEY"]
GOOGLE_SEARCH_ENGINE_ID = os.environ["GOOGLE_SEARCH_ENGINE_ID"]

openai_client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01",
)
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]

_trail_cache: dict[str, dict] = {}

HIGH_QUALITY_THRESHOLD = 7
MIN_RESULTS = 5
MAX_RESULTS = 15


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


async def geocode(client: httpx.AsyncClient, location: str) -> tuple[float, float]:
    resp = await client.get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={"address": location, "key": GOOGLE_PLACES_API_KEY},
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        raise HTTPException(status_code=404, detail=f"Location '{location}' not found")
    loc = results[0]["geometry"]["location"]
    return loc["lat"], loc["lng"]


async def search_trails(client: httpx.AsyncClient, location: str, lat: float, lng: float) -> list[dict]:
    resp = await client.get(
        "https://maps.googleapis.com/maps/api/place/textsearch/json",
        params={
            "query": f"hiking trails near {location}",
            "location": f"{lat},{lng}",
            "radius": 50000,
            "key": GOOGLE_PLACES_API_KEY,
        },
    )
    resp.raise_for_status()
    return resp.json().get("results", [])[:20]


async def get_place_details(client: httpx.AsyncClient, place_id: str) -> dict:
    resp = await client.get(
        "https://maps.googleapis.com/maps/api/place/details/json",
        params={
            "place_id": place_id,
            "fields": "name,rating,formatted_address,reviews,url,geometry",
            "key": GOOGLE_PLACES_API_KEY,
        },
    )
    resp.raise_for_status()
    return resp.json().get("result", {})


async def search_alltrails(client: httpx.AsyncClient, trail_name: str, location: str) -> str:
    resp = await client.get(
        "https://www.googleapis.com/customsearch/v1",
        params={
            "q": f'"{trail_name}" {location}',
            "key": GOOGLE_SEARCH_API_KEY,
            "cx": GOOGLE_SEARCH_ENGINE_ID,
            "num": 1,
        },
    )
    if resp.status_code != 200:
        return ""
    items = resp.json().get("items", [])
    return items[0].get("snippet", "") if items else ""


async def get_weather(client: httpx.AsyncClient, lat: float, lng: float) -> list[dict]:
    today = datetime.utcnow().date()
    days_until_saturday = (5 - today.weekday()) % 7 or 7
    saturday = today + timedelta(days=days_until_saturday)
    sunday = saturday + timedelta(days=1)
    # Include today + coming weekend so the AI can assess both current and upcoming conditions
    end_date = sunday if saturday != today else today + timedelta(days=1)

    resp = await client.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lng,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
            "temperature_unit": "fahrenheit",
            "windspeed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "auto",
            "start_date": str(today),
            "end_date": str(end_date),
        },
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    return [
        {
            "date": daily["time"][i],
            "high_f": daily["temperature_2m_max"][i],
            "low_f": daily["temperature_2m_min"][i],
            "precipitation_in": daily["precipitation_sum"][i],
            "max_wind_mph": daily["windspeed_10m_max"][i],
        }
        for i in range(len(daily.get("time", [])))
    ]


def synthesize(
    name: str,
    address: str,
    rating: float,
    distance_km: float,
    reviews: list[dict],
    alltrails_snippet: str,
    weather: list[dict],
) -> dict:
    weather_text = "\n".join(
        f"{d['date']}: High {d['high_f']}°F / Low {d['low_f']}°F, "
        f"Precipitation {d['precipitation_in']}in, Wind {d['max_wind_mph']}mph"
        for d in weather
    ) or "No forecast available."

    now_ts = datetime.utcnow().timestamp()
    annotated = []
    for r in reviews:
        age_days = (now_ts - r.get("time", 0)) / 86400 if r.get("time") else None
        relative = r.get("relative", "")
        if age_days is not None and age_days <= 14:
            prefix = f"[Recent: {relative}]"
        elif age_days is not None and 335 <= age_days <= 395:
            prefix = "[Same season, prior year]"
        else:
            prefix = f"[{relative}]" if relative else ""
        annotated.append(f"- {prefix} {r['text']}")
    reviews_text = "\n".join(annotated) or "No reviews available."
    alltrails_text = alltrails_snippet or "No AllTrails data available."

    prompt = f"""You are an expert hiking guide. Assess this trail's suitability for hiking TODAY.

Trail: {name} | {address} | Rating: {rating}/5 | Distance: {distance_km:.1f} km from city

Today's Weather (and upcoming weekend):
{weather_text}

Reviews (annotated by recency):
{reviews_text}

AllTrails snippet:
{alltrails_text}

Respond ONLY with valid JSON (no markdown fences):
{{
  "condition_summary": "2-3 sentences on trail conditions TODAY based on recent reviews and weather",
  "gear_list": ["item1", "item2", "item3", "item4"],
  "condition_tags": ["tag1", "tag2", "tag3"],
  "suitability_score": 7,
  "difficulty": "moderate"
}}

condition_tags: 3-6 short lowercase tags from: muddy, dry, wet, icy, dusty, crowded, quiet, shaded, exposed, windy, hot, cold, dog-friendly, well-maintained, overgrown, scenic

suitability_score: integer 0-10. 10=ideal conditions, 0=avoid today. Consider: recent rain→mud, high wind, extreme temps, very recent negative reviews.

difficulty: "easy", "moderate", or "hard" inferred from reviews/terrain.

gear_list: 4-6 specific items tailored to the conditions and weather."""

    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


async def fetch_trail_data(client: httpx.AsyncClient, trail: dict, location: str) -> dict:
    place_id = trail.get("place_id", "")
    name = trail.get("name", "Unknown Trail")

    details, alltrails_snippet = await asyncio.gather(
        get_place_details(client, place_id) if place_id else asyncio.sleep(0, result={}),
        search_alltrails(client, name, location),
    )

    reviews = [
        {
            "text": r["text"],
            "time": r.get("time", 0),
            "relative": r.get("relative_time_description", ""),
        }
        for r in details.get("reviews", []) if r.get("text")
    ]

    place_geo = details.get("geometry", {}).get("location", {})
    return {
        "trail": trail,
        "reviews": reviews,
        "alltrails_snippet": alltrails_snippet,
        "place_lat": place_geo.get("lat"),
        "place_lng": place_geo.get("lng"),
    }


@app.get("/get-trail-recommendations")
async def get_trail_recommendations(location: str = Query(..., description="City, state, or address")):
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    cache_key = f"{location.strip().lower()}:{date_str}"
    if cache_key in _trail_cache:
        return _trail_cache[cache_key]

    async with httpx.AsyncClient(timeout=30.0) as client:
        lat, lng = await geocode(client, location)
        trails, weather = await asyncio.gather(
            search_trails(client, location, lat, lng),
            get_weather(client, lat, lng),
        )
        trail_data = await asyncio.gather(*[
            fetch_trail_data(client, trail, location) for trail in trails
        ])

    scored = []
    for td in trail_data:
        trail = td["trail"]
        name = trail.get("name", "Unknown Trail")
        address = trail.get("formatted_address", "")
        rating = trail.get("rating", 0.0)
        place_id = trail.get("place_id", "")

        place_lat = td.get("place_lat") or trail.get("geometry", {}).get("location", {}).get("lat")
        place_lng = td.get("place_lng") or trail.get("geometry", {}).get("location", {}).get("lng")
        distance_km = round(haversine_km(lat, lng, place_lat, place_lng), 1) if place_lat and place_lng else None

        try:
            synthesis = synthesize(name, address, rating, distance_km or 0.0, td["reviews"], td["alltrails_snippet"], weather)
        except Exception as e:
            logger.error(f"Synthesis failed for {name}: {e}")
            synthesis = {
                "condition_summary": "Unable to generate condition summary at this time.",
                "gear_list": ["Water (2L+)", "Snacks", "Comfortable hiking shoes", "Sun protection"],
                "condition_tags": [],
                "suitability_score": 5,
                "difficulty": "moderate",
            }

        scored.append({
            "name": name,
            "address": address,
            "rating": rating,
            "distance_km": distance_km,
            "google_maps_url": f"https://www.google.com/maps/place/?q=place_id:{place_id}",
            "condition_summary": synthesis["condition_summary"],
            "gear_list": synthesis.get("gear_list", []),
            "condition_tags": synthesis.get("condition_tags", []),
            "suitability_score": synthesis.get("suitability_score", 5),
            "difficulty": synthesis.get("difficulty", "moderate"),
        })

    above = [t for t in scored if t["suitability_score"] >= HIGH_QUALITY_THRESHOLD]
    if len(above) >= MIN_RESULTS:
        result_trails = above[:MAX_RESULTS]
    else:
        result_trails = sorted(scored, key=lambda x: x["suitability_score"], reverse=True)[:MIN_RESULTS]

    response = {"trails": result_trails}
    _trail_cache[cache_key] = response
    return response


@app.get("/autocomplete")
async def autocomplete(input: str = Query(..., description="Search input for city autocomplete")):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://maps.googleapis.com/maps/api/place/autocomplete/json",
                params={
                    "input": input,
                    "types": "(cities)",
                    "key": GOOGLE_PLACES_API_KEY,
                },
            )
            resp.raise_for_status()
            predictions = resp.json().get("predictions", [])
            return {"suggestions": [p["description"] for p in predictions]}
    except Exception:
        return {"suggestions": []}


@app.get("/health")
async def health():
    return {"status": "ok"}
