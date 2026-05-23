import asyncio, sys
sys.path.append(".")
from backend.db.session import engine, Base
from backend.db import models  # noqa — registers models

async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created successfully.")

asyncio.run(main())