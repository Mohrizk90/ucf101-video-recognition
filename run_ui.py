#!/usr/bin/env python3
"""
Simple script to run the UCF101 Video Classification Web UI.
"""

import os
import sys
import subprocess
import webbrowser
import time

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import flask
        print("✅ Flask is installed")
    except ImportError:
        print("❌ Flask is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask>=2.3.0"])
        print("✅ Flask installed successfully")
    
    try:
        import torch
        print("✅ PyTorch is installed")
    except ImportError:
        print("❌ PyTorch is not installed")
        return False
    
    return True

def check_model_files():
    """Check if model files exist."""
    checkpoint_path = 'runs/ucf101_cnn_rnn_20250817_145949/best.pth'
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model checkpoint not found at: {checkpoint_path}")
        print("Please ensure you have trained the model first.")
        return False
    
    print(f"✅ Model checkpoint found: {checkpoint_path}")
    return True

def main():
    """Main function to run the web UI."""
    print("🎬 UCF101 Video Action Recognition Web UI")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    # Check model files
    if not check_model_files():
        print("❌ Model files not found")
        return
    
    print("\n🚀 Starting web server...")
    print("📱 The UI will open in your browser automatically")
    print("🌐 Server URL: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start the Flask app
        from app import app, load_model
        
        # Load model in background
        print("📦 Loading model...")
        load_model()
        print("✅ Model loaded successfully!")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://localhost:5000')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run the Flask app
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("Please check the error message above and try again.")

if __name__ == "__main__":
    main() 