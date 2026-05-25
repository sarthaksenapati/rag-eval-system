import sys
import traceback

print("=== STARTUP DEBUG ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Path: {sys.path}", flush=True)

try:
    print("Importing config...", flush=True)
    from backend.config import settings
    print(f"Config OK — collection: {settings.collection_name}", flush=True)

    print("Importing main...", flush=True)
    from backend import main
    print("Main OK", flush=True)

except Exception as e:
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("=== ALL IMPORTS OK ===", flush=True)