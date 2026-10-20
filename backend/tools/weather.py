"""
Live weather from Open-Meteo (free, no API key), used by web_search for weather questions.

A text search can't answer "what's the weather in X right now": result snippets are old or
generic pages, and a small model fills the gap with a made-up temperature.
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

WEATHER_QUESTION = re.compile(
    r"\b(weather|temperature|forecast|degrees|celsius|fahrenheit|raining|snowing|humid|humidity|windy)\b", re.IGNORECASE
)
# Words that aren't part of the place name.
_NOT_PLACE = set(
    "what whats what's is the weather temperature forecast degrees celsius fahrenheit in at for of right now today "
    "tonight tomorrow currently current like how hot cold it outside there there's search please tell me can could "
    "uh um okay ok so well hey metis also something else just know wanted ask "
    "you check look up find a an and or raining snowing humid humidity windy this week weekend "
    "no yes yeah give show get got need want tell say does do did will would should are was were be "
    "your my our their his her its going gonna pretty good thanks thank but".split()
)

_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
    82: "heavy showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


def _get(url: str) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "Metis/1.0"})
    with urllib.request.urlopen(request, timeout=8) as response:
        return json.loads(response.read())


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def place_words(query: str) -> List[str]:
    """The words of the query that could be a place name, in order."""
    words = [re.sub(r"'s$", "", w.strip(".'-")) for w in re.findall(r"[^\W\d_][\w'.-]*", query)]
    words = [w for w in words if w.lower() not in _NOT_PLACE and len(w) >= 3]
    # The transcriber capitalises names: in "No, just give me the weather for Moldova." only "Moldova" is a
    # place, and fuzzy-matching "give" found Givet, France.
    capitalised = [w for w in words if w[0].isupper()]
    return capitalised if capitalised and len(capitalised) < len(words) else words


def _geocode(name: str, country_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Best match for `name`; it must actually be called that (no 'Chichinovshchina' for 'Chichino')."""
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({"name": name, "count": 10})
    results = _get(url).get("results") or []
    wanted = _ascii(name)
    close = [r for r in results
             if difflib.SequenceMatcher(None, wanted, _ascii(r.get("name", ""))).ratio() >= 0.8
             and (country_code is None or r.get("country_code") == country_code)]
    # Countries and capitals first, then the most populated place.
    close.sort(key=lambda r: (r.get("feature_code") not in ("PCLI", "PPLC"), -(r.get("population") or 0)))
    return close[0] if close else None


def find_place(query: str) -> Optional[Dict[str, Any]]:
    words = place_words(query)
    if not words:
        return None
    # Try the whole phrase, then single words; a country in the query narrows down a town's name.
    candidates = [" ".join(words)] + ([w for w in words] if len(words) > 1 else [])
    places = [p for p in (_geocode(c) for c in dict.fromkeys(candidates)) if p]
    if not places:
        return None
    countries = [p for p in places if p.get("feature_code") == "PCLI"]
    towns = [p for p in places if p.get("feature_code") != "PCLI"]
    if countries and towns:
        in_country = [t for t in towns if t.get("country_code") == countries[0].get("country_code")]
        return in_country[0] if in_country else countries[0]
    return (towns or countries)[0]


def lookup(query: str) -> Optional[str]:
    """Current weather and the next two days for the place in `query`, or None if no place was found."""
    place = find_place(query)
    if not place:
        return None
    params = {
        "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto", "forecast_days": 2,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max",
    }
    data = _get("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params))
    now, daily = data["current"], data["daily"]
    where = place["name"] + (f", {place['country']}" if place.get("country") and place["country"] != place["name"] else "")
    if place.get("feature_code") == "PCLI":
        where += " (middle of the country)"
    lines = [
        f"Live weather for {where}, from Open-Meteo, at {now['time'].replace('T', ' ')} local time:",
        f"Now: {round(now['temperature_2m'])}°C (feels like {round(now['apparent_temperature'])}°C), "
        f"{_WMO.get(now['weather_code'], 'unknown conditions')}, humidity {now['relative_humidity_2m']}%, "
        f"wind {round(now['wind_speed_10m'])} km/h.",
    ]
    for i, label in enumerate(["Today", "Tomorrow"]):
        rain = daily.get("precipitation_probability_max", [None, None])[i]
        lines.append(f"{label}: {round(daily['temperature_2m_min'][i])} to {round(daily['temperature_2m_max'][i])}°C, "
                     f"{_WMO.get(daily['weather_code'][i], 'unknown conditions')}"
                     + (f", {rain}% chance of rain" if rain is not None else "") + ".")
    return "\n".join(lines)
