import sys
import traceback

print("=== STARTUP DEBUG ===", flush=True)

try:
    print("1. importing config...", flush=True)
    from backend.config import settings
    print("2. config OK", flush=True)

    print("3. importing FastAPI...", flush=True)
    from fastapi import FastAPI
    print("4. FastAPI OK", flush=True)

    print("5. importing main...", flush=True)
    from backend import main
    print("6. main OK", flush=True)

except Exception as e:
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("=== ALL OK ===", flush=True)