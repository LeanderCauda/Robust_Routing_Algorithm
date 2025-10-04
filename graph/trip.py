from graph.node import Node
from datetime import datetime


class Trip:
    def __init__(
        self, 
        source: Node, 
        target: Node,
        trip_id: str,
        trip_description: str,
        source_departure_time: datetime,
        target_arrival_time: datetime
    ):
        self.source = source
        self.target = target
        self.trip_id = trip_id
        self.trip_description = trip_description
        self.source_departure_time = source_departure_time
        self.target_arrival_time = target_arrival_time

    def is_walking(self) -> bool:
        return self.trip_id == "walking"

    def __repr__(self) -> str:
        return f"[{self.source.stop_name} ({self.source.stop_id})] [{self.source_departure_time.time()}] ---[{self.trip_description} ({self.trip_id})]---> [{self.target_arrival_time.time()}] [{self.target.stop_name} ({self.target.stop_id})]"

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.trip_id))

    def __eq__(self, other) -> bool:
        return self.source == other.source and self.target == other.target and self.trip_id == other.trip_id
