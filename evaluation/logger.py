"""
Simple logging utility for evaluation and analysis scripts
"""
import os
import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

class SimpleLogger:
    def __init__(self, script_name):
        self.script_name = script_name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(LOG_DIR, f"{script_name}_{timestamp}.log")
        self.start_time = datetime.datetime.now()
        
    def log(self, message, level="INFO"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        # Write to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
        
        # Print to console
        print(log_entry)
    
    def log_start(self):
        self.log(f"Starting {self.script_name}", "START")
        self.log(f"Log file: {self.log_file}")
    
    def log_complete(self):
        duration = (datetime.datetime.now() - self.start_time).total_seconds()
        self.log(f"Completed in {duration:.2f}s", "COMPLETE")
        self.log(f"Full log saved to: {self.log_file}")
