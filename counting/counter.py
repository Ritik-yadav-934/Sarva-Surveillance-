import numpy as np
import supervision as sv

class ZoneCounter:

    def __init__(self):

        self.polygon = np.array([
            [400, 200],
            [1000, 200],
            [1000, 700],
            [400, 700]
        ])

        self.zone = sv.PolygonZone(
        polygon=self.polygon
        )

        self.counted_ids = set()

        self.total_count = 0

    def process(self, detections):

        in_zone = self.zone.trigger(detections)

        for i, track_id in enumerate(detections.tracker_id):

            if in_zone[i]:

                if track_id not in self.counted_ids:

                    self.counted_ids.add(track_id)

                    self.total_count += 1

        return self.total_count