#!/bin/bash
# Quick start script for FastAPI + React application

echo "🚀 Starting RAG Document QA - Full Stack"
echo "========================================"
echo ""

# Check if Elasticsearch is running
if ! curl -s http://localhost:9200 > /dev/null; then
    echo "⚠️  Elasticsearch not running. Starting..."
    docker-compose up -d
    echo "✅ Elasticsearch started"
    echo "⏳ Waiting for Elasticsearch to be ready..."
    sleep 10
else
    echo "✅ Elasticsearch is running"
fi

echo ""
echo "📋 Starting services..."
echo ""

# Start backend in background
echo "🔧 Starting FastAPI backend on port 8000..."
cd backend
python api.py > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..
echo "   Backend PID: $BACKEND_PID"
echo "   Logs: backend.log"

# Wait for backend to start
sleep 3

# Start frontend in background
echo "🎨 Starting React frontend on port 3000..."
cd frontend
npm start > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "   Frontend PID: $FRONTEND_PID"
echo "   Logs: frontend.log"

echo ""
echo "✅ Services started successfully!"
echo ""
echo "🌐 Access the application:"
echo "   Frontend:  http://localhost:3000"
echo "   Backend:   http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
echo "📝 To stop services:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo "   Or press Ctrl+C and run: ./stop_fullstack.sh"
echo ""
echo "💡 Tip: Check logs with 'tail -f backend.log' or 'tail -f frontend.log'"
echo ""

# Save PIDs to file for cleanup
echo "$BACKEND_PID" > .pids
echo "$FRONTEND_PID" >> .pids

# Keep script running
wait

