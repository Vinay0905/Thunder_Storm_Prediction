import subprocess
import time
import sys
from pathlib import Path

# Path setup to ensure we run from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_servers():
    print("🚀 Launching the Thunderstorm Prediction System...")
    
    # 1. Start the FastAPI Server (Backend)
    # Using 'python -m uvicorn' is the most robust way to run it
    api_process = subprocess.Popen(
        ["uvicorn", "api.main:app", "--reload", "--port", "8000"],
        cwd=PROJECT_ROOT
    )
    print("✅ Backend API starting on http://localhost:8000")

    # Wait a second for the API to initialize
    time.sleep(2)

    # 2. Start the Streamlit App (Frontend)
    ui_process = subprocess.Popen(
        ["streamlit", "run", "streamlit/streamlit_ui.py", "--server.port", "8501"],
        cwd=PROJECT_ROOT
    )
    print("✅ Frontend UI starting on http://localhost:8501")

    print("\n--- System is running. Press Ctrl+C to stop both servers. ---")

    try:
        # Keep the script alive while both processes are running
        while True:
            time.sleep(1)
            # Check if either process has crashed
            if api_process.poll() is not None:
                print("⚠️ API Server stopped unexpectedly.")
                break
            if ui_process.poll() is not None:
                print("⚠️ Streamlit UI stopped unexpectedly.")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        api_process.terminate()
        ui_process.terminate()
        print("✅ Goodbye!")

if __name__ == "__main__":
    run_servers()
