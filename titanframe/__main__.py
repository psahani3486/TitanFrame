import argparse
import time
from titanframe.telemetry.server import start_dashboard, stop_dashboard

def main():
    parser = argparse.ArgumentParser(description="TitanFrame CLI")
    parser.add_argument("--dashboard", action="store_true", help="Start the real-time telemetry dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the dashboard on")
    
    args = parser.parse_args()
    
    if args.dashboard:
        start_dashboard(port=args.port)
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop_dashboard()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
