from src.processing import get_heights, get_times
import pandas as pd


class TestHeights:
    def test_zero(self):
        get_heights()
        assert True

    def test_one(self):
        h0, h1 = 200, 800
        heights = get_heights(min_height=h0, max_height=h1, resolution=1)
        assert len(heights) == h1-h0+1
    
    def test_two(self):
        h0, h1 = 200, 800
        heights = get_heights(min_height=h0, max_height=h1, resolution=2)
        assert len(heights) == (h1-h0) // 2 + 1

    def test_three(self):
        h0, h1 = 200, 800
        heights = get_heights(min_height=h0, max_height=h1)
        assert heights[0] == h0 and heights[-1] == h1


class TestTimes:
    def test_zero(self):
        get_times()
        assert True

    def test_one(self):
        year, month, day = "2023", "01", "01"
        times = get_times(pd.Timestamp(f"{year}-{month}-{day}"))
        assert times[0].day == int(day) and times[0].month == int(month) and times[0].year == int(year)

    def test_two(self):
        year, month, day = "2023", "01", "01"
        times = get_times(start_date=pd.Timestamp(f"{year}-{month}-{day}"))
        assert times[0].day == int(day) and times[0].month == int(month) and times[0].year == int(year) and times[0].hour == 19
    
    def test_three(self):
        year, month, day = "2023", "12", "31"
        times = get_times(pd.Timestamp(f"{year}-{month}-{day}"))
        next_day = pd.Timestamp(f"{year}-{month}-{day}") + pd.Timedelta(19 + 12, unit='hours')
        assert times[-1].day == next_day.day and times[-1].month == next_day.month and times[-1].year == next_day.year and times[-1].hour == next_day.hour
    