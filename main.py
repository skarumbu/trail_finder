import asyncio
import json
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

import httpx
from openai import AzureOpenAI
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

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
    return resp.json().get("results", [])[:5]


async def get_place_details(client: httpx.AsyncClient, place_id: str) -> dict:
    resp = await client.get(
        "https://maps.googleapis.com/maps/api/place/details/json",
        params={
            "place_id": place_id,
            "fields": "name,rating,formatted_address,reviews,url",
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


async def get_weekend_weather(client: httpx.AsyncClient, lat: float, lng: float) -> list[dict]:
    today = datetime.utcnow().date()
    days_until_saturday = (5 - today.weekday()) % 7 or 7
    saturday = today + timedelta(days=days_until_saturday)
    sunday = saturday + timedelta(days=1)

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
            "start_date": str(saturday),
            "end_date": str(sunday),
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
    google_reviews: list[str],
    alltrails_snippet: str,
    weather: list[dict],
) -> dict:
    weather_text = "\n".join(
        f"{d['date']}: High {d['high_f']}°F / Low {d['low_f']}°F, "
        f"Precipitation {d['precipitation_in']}in, Wind {d['max_wind_mph']}mph"
        for d in weather
    ) or "No forecast available."

    reviews_text = "\n".join(f"- {r}" for r in google_reviews) or "No reviews available."
    alltrails_text = alltrails_snippet or "No AllTrails data available."

    prompt = f"""You are an expert hiking guide. Based on the data below, write a trail condition summary and gear list for hiking this trail this weekend.

Trail: {name}
Location: {address}
Google Maps Rating: {rating}/5

Weekend Weather Forecast:
{weather_text}

Recent Google Maps Reviews:
{reviews_text}

AllTrails Trip Report:
{alltrails_text}

Respond ONLY with valid JSON in this exact format (no markdown fences):
{{
  "condition_summary": "2-3 sentences on current trail conditions based on reviews and weather",
  "gear_list": ["item1", "item2", "item3", "item4"]
}}

The gear_list should have 4-6 specific items tailored to the conditions and weather."""

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

    return {
        "trail": trail,
        "reviews": [r["text"] for r in details.get("reviews", []) if r.get("text")],
        "alltrails_snippet": alltrails_snippet,
    }


@app.get("/get-trail-recommendations")
async def get_trail_recommendations(location: str = Query(..., description="City, state, or address")):
    cache_key = location.strip().lower()
    if cache_key in _trail_cache:
        entry = _trail_cache[cache_key]
        if datetime.utcnow() - entry["timestamp"] < timedelta(hours=24):
            return entry["response"]

    async with httpx.AsyncClient(timeout=30.0) as client:
        lat, lng = await geocode(client, location)
        trails, weather = await asyncio.gather(
            search_trails(client, location, lat, lng),
            get_weekend_weather(client, lat, lng),
        )
        trail_data = await asyncio.gather(*[
            fetch_trail_data(client, trail, location) for trail in trails
        ])

    results = []
    for td in trail_data:
        trail = td["trail"]
        name = trail.get("name", "Unknown Trail")
        address = trail.get("formatted_address", "")
        rating = trail.get("rating", 0.0)
        place_id = trail.get("place_id", "")

        try:
            synthesis = synthesize(name, address, rating, td["reviews"], td["alltrails_snippet"], weather)
        except Exception as e:
            print(f"Synthesis failed for {name}: {e}")
            synthesis = {
                "condition_summary": "Unable to generate condition summary at this time.",
                "gear_list": ["Water (2L+)", "Snacks", "Comfortable hiking shoes", "Sun protection"],
            }

        results.append({
            "name": name,
            "address": address,
            "rating": rating,
            "google_maps_url": f"https://www.google.com/maps/place/?q=place_id:{place_id}",
            "condition_summary": synthesis["condition_summary"],
            "gear_list": synthesis["gear_list"],
        })

    response = {"trails": results}
    _trail_cache[cache_key] = {"response": response, "timestamp": datetime.utcnow()}
    return response


@app.get("/health")
async def health():
    return {"status": "ok"}
