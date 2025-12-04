"""
places_updater.py

Updates 'places_database.csv' with event counts for the CURRENT CALENDAR WEEK (Mon-Sun).
This ensures indices match the populartimes format (Index 0 = Monday).

1. Fetches live agenda from OpenData BCN.
2. Filters events occurring this week.
3. Counts intersections (100m radius) for each day.
4. Saves column 'weekly_events' as JSON array: [Mon_Count, Tue_Count, ... Sun_Count]
"""

import pandas as pd
import geopandas as gpd
import requests
import datetime
import json
import csv
import sys

# --- CONFIGURATION ---
INPUT_FILE = "places_database.csv"
BCN_AGENDA_URL = "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search?resource_id=877ccf66-9106-4ae2-be51-95a9f6469e4c&limit=15000"
VICINITY_METERS = 100 

def get_current_week_dates():
    """
    Returns a list of 7 dates (YYYY-MM-DD) for the current week, 
    ALWAYS starting on MONDAY.
    """
    today = datetime.date.today()
    # Find the Monday of the current week
    start_of_week = today - datetime.timedelta(days=today.weekday())
    
    week_dates = []
    print(f"🗓️  Calendar Week: {start_of_week} (Mon) to {start_of_week + datetime.timedelta(days=6)} (Sun)")
    
    for i in range(7):
        day = start_of_week + datetime.timedelta(days=i)
        week_dates.append(day)
        
    return week_dates

def main():
    print("\n--- 📡 STARTING EVENT SYNC ---")

    # 1. LOAD PLACES
    try:
        df_places = pd.read_csv(INPUT_FILE)
        
        # Convert to GeoDataFrame for spatial math
        gdf_places = gpd.GeoDataFrame(
            df_places,
            geometry=gpd.points_from_xy(df_places.longitude, df_places.latitude),
            crs="EPSG:4326"
        )
        # Project to Meters (EPSG:25831) for accurate 100m buffering
        gdf_places_meters = gdf_places.to_crs("EPSG:25831")
        gdf_places_meters['geometry'] = gdf_places_meters.geometry.buffer(VICINITY_METERS)
        
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    # 2. FETCH EVENTS
    print("📥 Downloading Agenda from OpenData BCN...")
    try:
        resp = requests.get(BCN_AGENDA_URL)
        resp.raise_for_status()
        records = resp.json()["result"]["records"]
        df_events = pd.DataFrame(records)
        
        # COLUMN MAPPING (As requested)
        # Coordinates
        lat_col = 'geo_epgs_4326_lat' if 'geo_epgs_4326_lat' in df_events.columns else 'lat'
        lon_col = 'geo_epgs_4326_lon' if 'geo_epgs_4326_lon' in df_events.columns else 'lon'
        
        # Dates
        start_col = 'start_date'
        end_col = 'end_date'

        if start_col not in df_events.columns:
            # Fallback if API changes names, but prioritizing user request
            print(f"⚠️ Warning: '{start_col}' not found. Columns: {list(df_events.columns)}")
            return

        # Clean & Convert
        df_events['lat'] = pd.to_numeric(df_events[lat_col], errors='coerce')
        df_events['lon'] = pd.to_numeric(df_events[lon_col], errors='coerce')
        df_events = df_events.dropna(subset=['lat', 'lon'])

        # Convert to Date Objects
        df_events['start_dt'] = pd.to_datetime(df_events[start_col], errors='coerce').dt.date
        
        if end_col in df_events.columns:
            df_events['end_dt'] = pd.to_datetime(df_events[end_col], errors='coerce').dt.date
            df_events['end_dt'] = df_events['end_dt'].fillna(df_events['start_dt'])
        else:
            df_events['end_dt'] = df_events['start_dt']

        # Convert Events to GeoDataFrame (Meters)
        gdf_events = gpd.GeoDataFrame(
            df_events,
            geometry=gpd.points_from_xy(df_events.lon, df_events.lat),
            crs="EPSG:4326"
        ).to_crs("EPSG:25831")
        
        print(f"✅ Parsed {len(gdf_events)} valid events.")

    except Exception as e:
        print(f"❌ API Error: {e}")
        return

    # 3. COMPUTE INTERSECTIONS PER DAY (Mon -> Sun)
    week_dates = get_current_week_dates()
    
    # Master list: [[Mon,Tue...], [Mon,Tue...]] aligned with places index
    all_schedules = [[] for _ in range(len(gdf_places))]

    for day_idx, current_date in enumerate(week_dates):
        day_name = current_date.strftime("%A")
        
        # Filter: Event must be active on this specific date
        active_events = gdf_events[
            (gdf_events['start_dt'] <= current_date) & 
            (gdf_events['end_dt'] >= current_date)
        ]
        
        count_overlaps = 0
        
        if len(active_events) > 0:
            # Spatial Join: Which places intersect which events?
            joined = gpd.sjoin(gdf_places_meters, active_events, how="left", predicate="intersects")
            
            # Count events per place index
            # dropna is important: 'left' join keeps places with 0 events as NaN rows
            valid_hits = joined.dropna(subset=['index_right'])
            counts = valid_hits.groupby(valid_hits.index).size()
            count_overlaps = len(valid_hits)

            for idx in range(len(gdf_places)):
                val = counts.get(idx, 0)
                all_schedules[idx].append(int(val))
        else:
            # Zero events for everyone this day
            for sched in all_schedules:
                sched.append(0)

        print(f"   👉 {day_name} ({current_date}): {count_overlaps} local event impacts found.")

    # 4. EXPORT
    # Serialize to JSON string for CSV compatibility
    df_places['weekly_events'] = [json.dumps(sched) for sched in all_schedules]
    
    # Save with quoting enabled to handle the JSON strings safely
    df_places.to_csv(INPUT_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"💾 Updated {INPUT_FILE} with 'weekly_events'.")

if __name__ == "__main__":
    main()