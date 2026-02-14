import time
import os
import subprocess
import sys

def monitor():
    model_path = 'models/master_model_v1/best_model.keras'
    script_path = 'scripts/analyze_feature_importance.py'
    python_path = sys.executable 
    
    last_mtime = 0
    
    print(f"Monitoring {model_path} for updates...")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            if os.path.exists(model_path):
                mtime = os.path.getmtime(model_path)
                if mtime > last_mtime:
                    print(f"\n[UPDATE] Model updated at {time.ctime(mtime)}. Running analysis...")
                    subprocess.run([python_path, script_path, '--model', model_path])
                    last_mtime = mtime
            else:
                print(f"Waiting for model file {model_path} to be created...", end='\r')
            
            time.sleep(30) # Check every 30 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == '__main__':
    monitor()
