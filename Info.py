class Info:

    def __init__(self, time, timetick, FPOGX, FPOGY, FPOGD, FPOGV):
        self.time = time
        self.timetick = timetick
        self.FPOGX = FPOGX
        self.FPOGY = FPOGY
        self.FPOGD = FPOGD
        self.FPOGV = FPOGV

class RegionInfo:
    def __init__(self, x, y, region, time):
        self.x = x
        self.y = y
        self.region = region
        self.relativeTime = time

