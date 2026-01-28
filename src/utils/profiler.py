import time
import os
import csv
import psutil

class Profiler:
    _instance = None
    log_file = "outputs/cost_profile.csv"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Profiler, cls).__new__(cls)
            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            if not os.path.exists(cls.log_file):
                with open(cls.log_file, 'w') as f:
                    f.write("stage,duration_sec,memory_mb,tokens,video_name\n")
        return cls._instance

    def log(self, stage, duration, tokens=0, video="unknown"):
        mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        with open(self.log_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([stage, f"{duration:.4f}", f"{mem:.2f}", tokens, video])

profiler = Profiler()
