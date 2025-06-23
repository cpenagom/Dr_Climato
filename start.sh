#!/bin/bash

echo "🚀 Starting LLM Chatbot Setup..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed!"
    echo "Please install Ollama first:"
    echo "curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

echo "✅ Ollama found"

# Check if Ollama service is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "🔄 Starting Ollama service..."
    ollama serve &
    
    # Wait for Ollama to be ready
    echo "⏳ Waiting for Ollama to start..."
    sleep 5
    
    # Check again
    while ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
        echo "Still waiting for Ollama..."
        sleep 3
    done
fi

echo "✅ Ollama service is running"

# Check if llama3.2 model is available
if ! ollama list | grep -q "llama3.2"; then
    echo "📥 Pulling llama3.2 model (this may take a few minutes)..."
    ollama pull llama3.2
    echo "✅ Model downloaded"
else
    echo "✅ llama3.2 model already available"
fi

# Create and activate virtual environment
ENV_NAME="chatbot_env"
if [ ! -d "$ENV_NAME" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv $ENV_NAME
fi

echo "🔄 Activating virtual environment..."
source $ENV_NAME/bin/activate

# Install Python requirements in the virtual environment
echo "📦 Installing Python requirements in virtual environment..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🎉 Setup complete! Starting the chatbot..."
echo "🤖 Your chatbot is now running in virtual environment..."
python orch_10.py