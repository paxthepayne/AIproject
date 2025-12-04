"""
places_database_builder.py

Builds the master 'places_database.csv' by:
1. Fetching base POIs from Barcelona OpenData.
2. Enriching them with Google Maps data (ID, Name, Attributes).
3. Fetching granular Popular Times data.
4. Imputing missing crowd data using spatial nearest neighbors.

Output Columns: name, longitude, latitude, google_id, attributes, popular_times
"""

import requests
import pandas as pd
import googlemaps
import populartimes
import json
import sys
import time
import math
import numpy as np

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
GOOGLE_API_KEY = ""  
OUTPUT_FILE = "places_database.csv"
NEIGHBORS_TO_IMPUTE = 3 # For Step 4

# --------------------------------------------------------
# UTILS
# --------------------------------------------------------
def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    """
    if total == 0:
        return
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    if iteration == total: 
        sys.stdout.write('\n')
    sys.stdout.flush()

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance (in meters) between two points"""
    R = 6371e3 
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def parse_schedule(json_str):
    """Returns a numpy array (7 days x 24 hours) or None if empty"""
    try:
        data = json.loads(json_str)
        if not data or len(data) < 7: return None
        matrix = []
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        sorted_data = sorted(data, key=lambda x: days_order.index(x['name']) if x['name'] in days_order else 0)
        for day in sorted_data:
            matrix.append(day['data'])
        return np.array(matrix)
    except:
        return None

def format_schedule_back_to_json(numpy_matrix):
    """Converts numpy array back to JSON string"""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    output = []
    for i, day_name in enumerate(days):
        output.append({
            "name": day_name,
            "data": numpy_matrix[i].tolist()
        })
    return json.dumps(output)

def main():
    print(f"\n--- 🚀 STARTING BUILDER FOR {OUTPUT_FILE} ---")
    
    # --------------------------------------------------------
    # STEP 1: INITIALIZE BASE DATA (OpenData BCN)
    # --------------------------------------------------------
    print("\n[Step 1/4] Fetching Base POIs from OpenData BCN...")
    
    try:
        url = "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search?resource_id=31431b23-d5b9-42b8-bcd0-a84da9d8c7fa&limit=32000"
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()["result"]["records"]
        raw_df = pd.DataFrame(data)

        df = pd.DataFrame()
        df['name'] = raw_df['name'] 
        df['latitude'] = pd.to_numeric(raw_df['geo_epgs_4326_lat'], errors='coerce')
        df['longitude'] = pd.to_numeric(raw_df['geo_epgs_4326_lon'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])
        
        df['google_id'] = ""
        df['attributes'] = ""
        df['popular_times'] = ""

        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Step 1 Complete. Base database saved with {len(df)} locations.")

    except Exception as e:
        print(f"❌ Error in Step 1: {e}")
        return

    # --------------------------------------------------------
    # STEP 2: GOOGLE ENRICHMENT (ID, Name, Attributes)
    # --------------------------------------------------------
    print("\n[Step 2/4] Enriching with Google Maps (ID, Name, Attributes)...")
    
    gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
    df = pd.read_csv(OUTPUT_FILE)
    
    df['google_id'] = df['google_id'].astype('object')
    df['attributes'] = df['attributes'].astype('object')
    df['name'] = df['name'].astype('object')
    
    valid_rows = []
    total_rows = len(df)
    error_printed = False 

    for index, row in df.iterrows():
        original_name = str(row['name']).strip()
        search_query = original_name if "Barcelona" in original_name else f"{original_name}, Barcelona"
        
        try:
            find_res = gmaps.find_place(input=search_query, input_type="textquery")
            
            if find_res['status'] == 'OK' and len(find_res['candidates']) > 0:
                place_id = find_res['candidates'][0]['place_id']
                row['google_id'] = place_id
                
                try:
                    details_res = gmaps.place(place_id=place_id, fields=['name', 'type'])
                    if details_res['status'] == 'OK':
                        result = details_res['result']
                        row['name'] = result.get('name', original_name) 
                        row['attributes'] = json.dumps(result.get('types', [])) 
                except Exception as detail_err:
                    row['attributes'] = "[]"
                    if not error_printed:
                        print(f"\n⚠️ Warning on details fetch: {detail_err}")
                        error_printed = True

                valid_rows.append(row)
            else:
                pass 

        except Exception as e:
            if not error_printed:
                print(f"\n❌ CRITICAL API ERROR: {e}")
                error_printed = True
    
        print_progress(index + 1, total_rows, prefix='Progress:', suffix=f'Found: {len(valid_rows)}', length=40)

    df_enriched = pd.DataFrame(valid_rows)
    df_enriched.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Step 2 Complete. {len(valid_rows)} places matched & saved. ({total_rows - len(valid_rows)} dropped)")

    # --------------------------------------------------------
    # STEP 3: POPULAR TIMES FETCH
    # --------------------------------------------------------
    print("\n[Step 3/4] Fetching Popular Times (Scraping)...")
    
    df = pd.read_csv(OUTPUT_FILE)
    df['popular_times'] = df['popular_times'].astype('object')
    total_rows = len(df)
    success_count = 0
    
    for index, row in df.iterrows():
        place_id = str(row['google_id'])
        try:
            data = populartimes.get_id(GOOGLE_API_KEY, place_id)
            if 'populartimes' in data:
                row['popular_times'] = json.dumps(data['populartimes'])
                success_count += 1
            else:
                row['popular_times'] = "[]"
        except Exception:
            row['popular_times'] = "[]"

        df.iloc[index] = row
        print_progress(index + 1, total_rows, prefix='Progress:', suffix=f'Got Data: {success_count}', length=40)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Step 3 Complete. Popular times gathered for {success_count} places.")

    # --------------------------------------------------------
    # STEP 4: SPATIAL IMPUTATION (Fill Missing Data)
    # --------------------------------------------------------
    print("\n[Step 4/4] Imputing Missing Crowd Data (Spatial Neighbors)...")
    
    df = pd.read_csv(OUTPUT_FILE)
    
    # Identify Sources/Targets
    df['has_data'] = df['popular_times'].apply(lambda x: len(str(x)) > 20 if pd.notna(x) else False)
    sources = df[df['has_data']].copy()
    targets = df[~df['has_data']].copy()
    
    if len(sources) == 0:
        print("❌ Error: No source data found to spread! Step 3 failed.")
        return

    # Pre-parse sources
    source_schedules = {}
    for idx, row in sources.iterrows():
        sched = parse_schedule(row['popular_times'])
        if sched is not None:
            source_schedules[idx] = sched

    imputed_count = 0
    total_targets = len(targets)
    
    for idx, target in targets.iterrows():
        t_lat, t_lon = target['latitude'], target['longitude']
        distances = []
        
        # Distance to all sources
        for s_idx, source in sources.iterrows():
            dist = haversine(t_lat, t_lon, source['latitude'], source['longitude'])
            distances.append((s_idx, dist))
        
        # Top K Neighbors
        distances.sort(key=lambda x: x[1])
        nearest_k = distances[:NEIGHBORS_TO_IMPUTE]
        
        # Weighted Average
        weighted_sum = np.zeros((7, 24))
        total_weight = 0
        
        for s_idx, dist in nearest_k:
            weight = 1 / (max(dist, 50) ** 2) # Inverse Distance Weighting (Squared)
            sched = source_schedules.get(s_idx)
            if sched is not None:
                weighted_sum += (sched * weight)
                total_weight += weight
                
        if total_weight > 0:
            final_matrix = (weighted_sum / total_weight).astype(int)
            final_json = format_schedule_back_to_json(final_matrix)
            df.at[idx, 'popular_times'] = final_json
            imputed_count += 1
            
        print_progress(imputed_count, total_targets, prefix='Imputing:', length=40)

    # Cleanup and Save
    df.drop(columns=['has_data'], inplace=True)
    
    # IMPORTANT: Quote non-numeric fields to handle JSON strings properly
    df.to_csv(OUTPUT_FILE, index=False, quoting=1) 
    
    print(f"\n✅ Step 4 Complete. Filled {imputed_count} missing locations.")
    print(f"\n🎉 DATABASE BUILD COMPLETE. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()