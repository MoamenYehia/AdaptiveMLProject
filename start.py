"""
Helper script to run the Streamlit application
Run with: python start.py
"""
import subprocess
import sys
import os

def main():
    """Start the Streamlit application"""
    app_path = os.path.join(os.path.dirname(__file__), 'src', 'main.py')
    
    print("\n" + "="*50)
    print("Adaptive Fraud Detection System")
    print("="*50)
    print("\nStarting Streamlit application...")
    print("Opening at: http://localhost:8501\n")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_path
        ], check=True)
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
