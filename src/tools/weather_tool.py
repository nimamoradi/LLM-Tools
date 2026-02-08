# tools/weather_tool.py
from __future__ import annotations

import json
from typing import Any, Type
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.tools import BaseTool
import asyncio
import python_weather
from datetime import datetime


class WeatherArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location: str = Field(..., description="City/country or lat/lon")
    date: str = Field(..., description="YYYY-MM-DD")


class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Weather finder tool. Call get_weather(location, date) to get weather for that day for a single date and location only."
    args_schema: Type[BaseModel] = WeatherArgs

    def _run(self, location: str, date: str) -> str:
        return asyncio.run(self._arun(location=location, date=date))

    async def _arun(self, location: str, date: str) -> str:
        async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d").date()
                print(f"target_date: {target_date} for location: {location}" )
            except ValueError:
                return json.dumps({"ok": False, "error": "Invalid date format. Please use YYYY-MM-DD."})

            try:
                weather = await client.get(location)
                for daily in weather:
                    if daily.date == target_date:
                        forecast = {
                            'high': f"{daily.highest_temperature}°F",
                            'low': f"{daily.lowest_temperature}°F",
                            "sun_light": f"{daily.sunlight} Hours",
                            'rain': f"{daily.snowfall} Inch",
                            'location': location
                        }
                        return json.dumps({"ok": True, "location": location, "date": date, "forecast": forecast})

                return json.dumps({"ok": False, "error": f"No weather data found for {date} in {location}"})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})
