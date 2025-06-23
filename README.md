# Dr_Climato
# LLM Chatbot DEMO! Project

## Quick Start

### Prerequisites
1. **Install Ollama** (if not already installed):
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

### Setup and Run

#### Option 1: Automated Setup 
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Run the setup script (creates virtual env automatically)
chmod +x start.sh
./start.sh
```

#### Option 2: Manual Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Create virtual environment
python3 -m venv chatbot_env
source chatbot_env/bin/activate  # On Windows: chatbot_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Ollama with local API server (in another terminal)
ollama serve

# Pull the model (in another terminal)
ollama pull llama3.2

# Run llama3.2 locally (in another terminal) 
ollama run llama3.2

# Run the chatbot id10
python Or_id010.py

# Run the chatbot id15
python Or_id015.py
```

## File Structure
- `Or_id010.py` - Chatbot script
- `Or_id015.py` - Chatbot script
- `data/` - Your data files
- `requirements.txt` - Python dependencies
- `start.sh` - Automated setup script
- `chatbot_env/` - Virtual environment (created automatically)

## Notes
- The virtual environment `chatbot_env/` is created automatically by the start script
- Add `chatbot_env/` to your `.gitignore` so it doesn't get committed
- To deactivate the virtual environment later, just run `deactivate`

## Troubleshooting
- Make sure Ollama is running: `curl http://localhost:11434/api/tags`
- Check if model is available: `ollama list`
- If using manual setup, make sure virtual environment is activated: `which python`
