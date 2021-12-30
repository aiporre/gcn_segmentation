from unittest import TestCase

from lib.utils.timer import TimeTrack
import time


class TestTimeTrack(TestCase):
    def test_time_string_no_restart(self):
        time_track = TimeTrack(60)

        self.assertEqual(time_track.time_lap, None)
        self.assertGreater(time_track.start, 0)
        self.assertEqual(time_track.laps, 0)
        time.sleep(1)
        time_track.lap()
        self.assertEqual(time_track.laps, 1)
        # mocking time 1 seconds
        print(time_track.__str__())




