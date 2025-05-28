import asyncio
import aiohttp
import os
import json
import time

OSU_CLIENT_ID = os.getenv("OSU_CLIENT_ID")
OSU_CLIENT_SECRET = os.getenv("OSU_CLIENT_SECRET")
DOWNLOAD_DIR = "../osu_beatmaps"
MAX_BEATMAPS_TO_COLLECT = 2000           
MAX_CONCURRENT_DOWNLOADS = 10             
BEATMAP_DOWNLOAD_URL_TEMPLATE = "https://dl.sayobot.cn/beatmaps/download/novideo/{set_id}"

OSU_API_TOKEN_URL = "https://osu.ppy.sh/oauth/token"
OSU_API_SEARCH_URL = "https://osu.ppy.sh/api/v2/beatmapsets/search"

_session = None
_api_token = None
_token_expiry_time = 0 
_token_lock = asyncio.Lock()

# filter settings
low_star_limit = 5.0
high_star_limit = 9.0
low_ar_limit = 9.0
high_ar_limit = 10.0


async def ensure_api_token():
    global _api_token, _token_expiry_time
    async with _token_lock:
        # check token validity
        if _api_token and time.time() < _token_expiry_time - 300:
            return _api_token

        print("Acquiring/Refreshing osu! API token...")
        payload = {
            "client_id": OSU_CLIENT_ID,
            "client_secret": OSU_CLIENT_SECRET,
            "grant_type": "client_credentials",
            "scope": "public",
        }
        async with _session.post(OSU_API_TOKEN_URL, data=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Failed to get API token: {response.status} - {error_text}")
            token_data = await response.json()
            _api_token = token_data["access_token"]
            _token_expiry_time = time.time() + token_data.get("expires_in", 3600) 
            print("API Token acquired/refreshed.")
            return _api_token

async def search_beatmaps_page(query, mode="osu", status="ranked", cursor_string=None, sort_by=None):
    token = await ensure_api_token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    params = {"q": query, "m": mode, "s": status, "nsfw": "false"}
    if cursor_string:
        params["cursor_string"] = cursor_string
    if sort_by: 
        params["sort"] = sort_by

    for attempt in range(3):
        try:
            async with _session.get(OSU_API_SEARCH_URL, headers=headers, params=params) as response:
                if response.status == 401: 
                    print("API Token unauthorized (401), forcing refresh...")
                    await ensure_api_token() 
                    if attempt < 2 : continue 
                    else: raise Exception("Failed to authorize after token refresh.")

                if response.status == 429: 
                    retry_after = int(response.headers.get("Retry-After", 30))
                    print(f"API rate limit hit. Waiting for {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                    continue 

                response.raise_for_status() 
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"Network error during search (attempt {attempt+1}/3): {e}")
            if attempt == 2: raise
            await asyncio.sleep(5 + attempt * 5) 
    return None 

def filter_beatmapset(bms_data):
    
    if bms_data.get("status") not in ["ranked"]: 
        return False
    
    if bms_data.get("play_count", 0) < 1000:
        return False

    for diff in bms_data.get("beatmaps", []):
        stars = diff.get("difficulty_rating", 0)
        ar = diff.get("ar", 0)
        # cs = diff.get("cs", 0)
        # od = diff.get("accuracy", 0) 
        # hp = diff.get("drain", 0)   
        # bpm = diff.get("bpm", 0)
        # length_seconds = diff.get("total_length", 0)

        if low_star_limit <= stars <= high_star_limit and low_ar_limit <= ar <= high_ar_limit:
            return True 

    return False 

async def download_beatmap_osz(set_id, download_url_template, target_dir):
    filepath = os.path.join(target_dir, f"{set_id}.osz")
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return set_id, "skipped"

    download_url = download_url_template.format(set_id=set_id)
    try:
        timeout = aiohttp.ClientTimeout(total=300) # set timeout (secs)
        async with _session.get(download_url, timeout=timeout) as response:
            if response.status == 200:
                with open(filepath, "wb") as f:
                    while True:
                        chunk = await response.content.read(1024 * 1024) # 1MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                return set_id, "downloaded"
            else:
                return set_id, f"error_{response.status}"
    except asyncio.TimeoutError:
        return set_id, "timeout"
    except Exception as e:
        return set_id, f"exception_{type(e).__name__}"


async def main():
    global _session
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    connector = aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_DOWNLOADS + 5) 
    async with aiohttp.ClientSession(connector=connector) as session:
        _session = session 
        await ensure_api_token()

        collected_beatmap_sets_metadata = [] 
        collected_set_ids = set() 
        rating_table = {}
        current_cursor = None
        total_api_searches = 0

        search_api_query = "stars>={low_star_limit} stars<={high_star_limit} mode=osu status=ranked" 
        search_api_sort = "favourites_desc"

        print(f"Starting to collect up to {MAX_BEATMAPS_TO_COLLECT} beatmaps with query: '{search_api_query}'")
        print(f"Filtering criteria: {low_star_limit} <= stars <= {high_star_limit} AND {high_ar_limit} >= AR >= {low_ar_limit}")

        while len(collected_set_ids) < MAX_BEATMAPS_TO_COLLECT:
            total_api_searches += 1
            print(f"\nAPI Search #{total_api_searches}, Collected {len(collected_set_ids)}/{MAX_BEATMAPS_TO_COLLECT} sets. Cursor: {current_cursor}")
            page_data = await search_beatmaps_page(
                search_api_query,
                cursor_string=current_cursor,
                sort_by=search_api_sort
            )

            if not page_data or not page_data.get("beatmapsets"):
                print("No more beatmapsets found from API or error fetching page.")
                break

            newly_found_on_page = 0
            for bms in page_data["beatmapsets"]:
                if bms["id"] not in collected_set_ids:
                    if filter_beatmapset(bms): 
                        collected_set_ids.add(bms["id"])
                        collected_beatmap_sets_metadata.append(bms)
                        rating_table[bms["id"]] = bms.get("rating", 0)
                        newly_found_on_page += 1
                        if len(collected_set_ids) >= MAX_BEATMAPS_TO_COLLECT:
                            break
            print(f"Accepted {newly_found_on_page} new sets from this page.")

            if len(collected_set_ids) >= MAX_BEATMAPS_TO_COLLECT:
                print(f"Reached target of {MAX_BEATMAPS_TO_COLLECT} beatmaps.")
                break

            current_cursor = page_data.get("cursor_string")
            if not current_cursor:
                print("Reached end of API search results.")
                break

            await asyncio.sleep(1.2) 

        print(f"\n--- Collection Phase Complete ---")
        print(f"Collected {len(collected_set_ids)} unique beatmap sets for download.")


        if collected_beatmap_sets_metadata:
            metadata_filepath = os.path.join(DOWNLOAD_DIR, "collected_metadata.json")
            final_metadata_list = []
            action_taken_metadata = "saved"

            if os.path.exists(metadata_filepath):
                try:
                    with open(metadata_filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    if content.strip(): 
                        data = json.loads(content)
                        if isinstance(data, list):
                            final_metadata_list = data
                            action_taken_metadata = "appended to"
                        else:
                            print(f"Warning: Existing metadata file {metadata_filepath} does not contain a list. It will be overwritten.")
                            action_taken_metadata = "overwritten in"
                    else: 
                        action_taken_metadata = "saved to"
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {metadata_filepath}. File will be overwritten.")
                    action_taken_metadata = "overwritten in"
                except Exception as e:
                    print(f"Warning: Error reading {metadata_filepath}: {e}. File will be overwritten.")
                    action_taken_metadata = "overwritten in"
                
                final_metadata_list.extend(collected_beatmap_sets_metadata) 
                
                with open(metadata_filepath, "w", encoding="utf-8") as f:
                    json.dump(final_metadata_list, f, ensure_ascii=False, indent=2)
            else:
                with open(metadata_filepath, "w", encoding="utf-8") as f:
                    json.dump(collected_beatmap_sets_metadata, f, ensure_ascii=False, indent=2)
                action_taken_metadata = "saved to"

        if collected_beatmap_sets_metadata: 
            print(f"Collected metadata {action_taken_metadata}: {metadata_filepath}")

        if rating_table:
            rating_filepath = os.path.join(DOWNLOAD_DIR, "rating_table.json")
            final_rating_dict = {}
            action_taken_rating = "saved" 

            if os.path.exists(rating_filepath):
                try:
                    with open(rating_filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    if content.strip(): 
                        data = json.loads(content)
                        if isinstance(data, dict):
                            final_rating_dict = data
                            action_taken_rating = "updated in"
                        else:
                            print(f"Warning: Existing rating file {rating_filepath} does not contain a dictionary. It will be overwritten.")
                            action_taken_rating = "overwritten in"
                    else: 
                        action_taken_rating = "saved to"
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {rating_filepath}. File will be overwritten.")
                    action_taken_rating = "overwritten in"
                except Exception as e:
                    print(f"Warning: Error reading {rating_filepath}: {e}. File will be overwritten.")
                    action_taken_rating = "overwritten in"
                final_rating_dict.update(rating_table)         
                with open(rating_filepath, "w", encoding="utf-8") as f:
                    json.dump(final_rating_dict, f, ensure_ascii=False, indent=2)
            else:
                with open(rating_filepath, "w", encoding="utf-8") as f:
                    json.dump(rating_table, f, ensure_ascii=False, indent=2)
                action_taken_rating = "saved to"    
                
        if rating_table: 
            print(f"Rating table {action_taken_rating}: {rating_filepath}")

        if not collected_set_ids:
            print("No beatmaps to download.")
            return

        download_tasks = []
        download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

        async def guarded_download(set_id):
            async with download_semaphore:
                return await download_beatmap_osz(set_id, BEATMAP_DOWNLOAD_URL_TEMPLATE, DOWNLOAD_DIR)

        print(f"\nStarting download of {len(collected_set_ids)} beatmaps with {MAX_CONCURRENT_DOWNLOADS} concurrent workers...")
        download_start_time = time.time()

        for set_id_to_download in collected_set_ids:
            download_tasks.append(asyncio.create_task(guarded_download(set_id_to_download)))

        download_results = []
        for i, task_future in enumerate(asyncio.as_completed(download_tasks)):
            result = await task_future
            download_results.append(result)
            if (i + 1) % 20 == 0 or (i + 1) == len(download_tasks): 
                elapsed = time.time() - download_start_time
                print(f"Download progress: {i+1}/{len(download_tasks)} ({(i+1)*100/len(download_tasks):.2f}%) completed. Time: {elapsed:.2f}s")

        succeeded = sum(1 for _, status in download_results if status == "downloaded")
        skipped = sum(1 for _, status in download_results if status == "skipped")
        failed = len(download_results) - succeeded - skipped

        print("\n--- Download Phase Complete ---")
        print(f"Successfully downloaded: {succeeded}")
        print(f"Skipped (already existed): {skipped}")
        print(f"Failed (errors/timeouts/not found): {failed}")
        total_download_time = time.time() - download_start_time
        print(f"Total download time: {total_download_time:.2f} seconds.")

if __name__ == "__main__":
    if OSU_CLIENT_ID == "YOUR_OSU_CLIENT_ID" or OSU_CLIENT_SECRET == "YOUR_OSU_CLIENT_SECRET":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!  OSU_CLIENT_ID and OSU_CLIENT_SECRET missing !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        asyncio.run(main())