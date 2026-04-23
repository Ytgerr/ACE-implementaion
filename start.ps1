# Start backend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd src; uv run uvicorn main:app"

# Start frontend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd src/frontend; npm install; npm run dev"