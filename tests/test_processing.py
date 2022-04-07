from src.processing import get_heights, get_times, get_idx
import pandas as pd
import numpy as np


class TestHeights:
    def test_default_params(self):
        get_heights()
        assert True

    def test_length(self):
        h0, h1 = 200, 800
        heights = get_heights(min_height=h0, max_height=h1, resolution=1)
        assert len(heights) == h1-h0+1
    
    def test_length_with_resolution(self):
        h0, h1 = 200, 800
        heights = get_heights(min_height=h0, max_height=h1, resolution=2)
        assert len(heights) == (h1-h0) // 2 + 1

    def test_boundaries(self):
        h0, h1 = 200, 800
        heights = get_heights(min_height=h0, max_height=h1)
        assert heights[0] == h0 and heights[-1] == h1


class TestTimes:
    def test_default_params(self):
        get_times()
        assert True

    def test_date(self):
        year, month, day = "2023", "01", "01"
        times = get_times(pd.Timestamp(f"{year}-{month}-{day}"))
        assert times[0].day == int(day) and times[0].month == int(month) and times[0].year == int(year)

    def test_hour(self):
        year, month, day = "2023", "01", "01"
        times = get_times(start_date=pd.Timestamp(f"{year}-{month}-{day}"))
        assert times[0].day == int(day) and times[0].month == int(month) and times[0].year == int(year) and times[0].hour == 19
    
    def test_boundaries(self):
        year, month, day = "2023", "12", "31"
        times = get_times(pd.Timestamp(f"{year}-{month}-{day}"))
        next_day = pd.Timestamp(f"{year}-{month}-{day}") + pd.Timedelta(19 + 12, unit='hours')
        assert times[-1].day == next_day.day and times[-1].month == next_day.month and times[-1].year == next_day.year and times[-1].hour == next_day.hour
    

class TestIndices:
    def test_heights(self):
        heights = get_heights()
        h1, hn = heights[0], heights[-1]
        delta = heights[1] - h1
        h1 += delta / 2
        hn -= delta / 2
        query_heights = np.linspace(h1, hn, len(heights)-1)
        idxs = [ get_idx(x=query_height, arr=heights) for query_height in query_heights ]
        assert np.arange(len(heights)-1).tolist() == idxs

    def test_times(self):
        times = get_times()
        t1, tn = times[0], times[-1]
        delta = times[1] - t1
        t1 += delta / 2
        tn -= delta / 2
        query_times = pd.to_datetime(np.linspace(t1.value, tn.value, len(times)-1, dtype=np.int64))
        idxs = [ get_idx(x=query_time, arr=times) for query_time in query_times ]
        assert np.arange(len(times)-1).tolist() == idxs
