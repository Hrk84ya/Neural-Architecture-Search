#!/usr/bin/env python3
"""
Neural Architecture Search Launcher
Simple launcher for the NAS GUI
"""

import webbrowser
import time
import threading

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://localhost:5001')

if __name__ == "__main__":
    try:
        from gui import app
        
        print("🚀 Starting Neural Architecture Search...")
        print("🌐 GUI will open at: http://localhost:5001")
        
        # Open browser
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start app
        app.run(debug=False, host='0.0.0.0', port=5001)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("📦 Install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")