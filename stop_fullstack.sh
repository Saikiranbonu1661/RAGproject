#!/bin/bash
# Stop all services

echo "🛑 Stopping RAG Document QA services..."

if [ -f .pids ]; then
    while read pid; do
        if kill -0 $pid 2>/dev/null; then
            echo "   Stopping PID: $pid"
            kill $pid
        fi
    done < .pids
    rm .pids
    echo "✅ Services stopped"
else
    echo "⚠️  No .pids file found. Stopping by port..."
    
    # Kill backend (port 8000)
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    
    # Kill frontend (port 3000)
    lsof -ti:3000 | xargs kill -9 2>/dev/null
    
    echo "✅ Ports cleared"
fi

# Optionally stop Elasticsearch
read -p "Stop Elasticsearch? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose down
    echo "✅ Elasticsearch stopped"
fi

echo "👋 All done!"

