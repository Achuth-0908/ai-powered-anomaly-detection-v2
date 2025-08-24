import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    
    print("🚀 Starting AI Anomaly Detection System...")
    print("📊 Launching Streamlit interface...")
    
    # Get the directory of this script and find the main app
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Look for streamlit_app.py in parent directory (root)
    parent_dir = os.path.dirname(script_dir)
    streamlit_app_path = os.path.join(parent_dir, "streamlit_app.py")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            streamlit_app_path,
            "--browser.gatherUsageStats", "false"
        ], check=True)

    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Streamlit application...")
        sys.exit(0)

if __name__ == "__main__":
    main()


