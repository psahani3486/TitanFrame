"""
TitanFrame eCommerce Web Dashboard Launcher
=============================================
Launches the non-blocking TitanFrame dashboard server, populates live telemetry,
and keeps the HTTP server responding smoothly to the web interface.
"""

import time
import os
import threading
import titanframe as tf

def launch_dashboard(port: int = 8080):
    print("=" * 65)
    print("      TitanFrame Web Dashboard & Telemetry Launcher      ")
    print("=" * 65)
    
    # 1. Start the HTTP Dashboard server (runs server in background daemon thread)
    tf.start_dashboard(port=port)
    print(f"\n[+] Live Dashboard running at: http://localhost:{port}")
    print(f"[+] Live REST API available at: http://localhost:{port}/api/metrics")
    print(f"[+] Datasets API available at: http://localhost:{port}/api/datasets")
    
    # Set default config settings
    tf.config.spill_threshold = 0.85
    tf.config.chunk_size = 65536
    tf.config.enable_query_optimizer = True
    
    print(f"\n[!] Dashboard Server is LIVE & NON-BLOCKING on http://localhost:{port}")
    print("    Visit http://localhost:8080 in your browser to interact with the Studio.")
    print("    Press Ctrl+C to terminate.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        tf.stop_dashboard()
        print("Dashboard server stopped.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    launch_dashboard(port=port)
