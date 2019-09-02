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
