#!/usr/bin/env bash

# Start backend
cd src || exit
uv run uvicorn main:app &
BACK_PID=$!

# Start frontend
cd frontend || exit
npm run dev &
FRONT_PID=$!

echo "Backend PID: $BACK_PID"
echo "Frontend PID: $FRONT_PID"

# Wait for both processes
wait $BACK_PID $FRONT_PID