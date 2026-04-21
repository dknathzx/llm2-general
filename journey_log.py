import json, os
from datetime import datetime

JOURNEY_FILE = "/kaggle/working/journey_log.json"

def log(action, status, details="", error=""):
    entry = {
        "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action"    : action,
        "status"    : status,
        "details"   : details,
        "error"     : error
    }
    data = {"journey": []}
    if os.path.exists(JOURNEY_FILE):
        with open(JOURNEY_FILE) as f:
            data = json.load(f)
    data["journey"].append(entry)
    with open(JOURNEY_FILE, "w") as f:
        json.dump(data, f, indent=2)
    icon = {"OK":"✅","ERROR":"❌",
            "RUNNING":"🔄","SKIPPED":"⏭️"}.get(status,"📝")
    print(f"{icon} [{entry['timestamp']}] {action} — {status}")
    if details: print(f"   {details}")
    if error:   print(f"   ERROR: {error}")
