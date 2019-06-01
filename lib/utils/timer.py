import time

class Timer():
    def __init__(self, time_threshold):
        self.start = time.time()
        self.time_threshold = time_threshold

    def is_time(self):
        current_time = time.time()
        time_diff = current_time-self.start
        if time_diff>self.time_threshold:
            print("time elapsed")
            print(time_diff)
            self.start = current_time
            return True
        else:
            return False


class TimeTrack:
    def __init__(self, total_laps=None):
        self.start = time.time()
        self.current_time = self.start
        self.time_lap = None
        self.total_laps = total_laps
        self.laps = 0

    def restart(self):
        self.start = time.time()
        self.current_time= self.start
        self.laps = 0
        self.time_lap = self.start

    def lap(self):
        current_time = time.time()
        time_diff = current_time-self.current_time
        self.current_time = current_time
        self.time_lap = time_diff
        self.laps += 1

    def get_completion_time(self):
        # computes the final given the last lap time
        laps_missing = self.total_laps - self.laps
        estimate_time = self.time_lap*laps_missing if self.time_lap is not None else None
        return estimate_time

    def __str__(self):
        completion_time = self.get_completion_time()
        completion_time = time.strftime('%H:%M:%S', time.gmtime(completion_time))
        total_lapsed_time = self.current_time - self.start
        total_lapsed_time = time.strftime('%H:%M:%S', time.gmtime(total_lapsed_time))
        time_str = f"{self.time_lap:.2f}/it, E={completion_time} T={total_lapsed_time}"
        return time_str