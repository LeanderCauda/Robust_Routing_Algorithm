from graph.node import Node
from datetime import datetime


class Edge:
    def __init__(
        self,
        source: Node,
        target: Node,
        trip_id: str,
        trip_description: str,
        transfer_time: int,
        source_arrival_time: datetime | int,
        target_departure_time: datetime | int
    ):
        self.source = source
        self.target = target
        self.trip_id = trip_id
        self.trip_description = trip_description
        self.transfer_time = transfer_time
        
        if self.is_walking():
            self.source_arrival_time   = source_arrival_time
            self.target_departure_time = 0
        else:
            self.source_arrival_time   = source_arrival_time
            self.target_departure_time = target_departure_time

    def is_walking(self) -> bool:
        return self.trip_id == "walking"
    
    def __repr__(self) -> str:
        if self.is_walking():
            return f"[Walking] [{self.source_arrival_time}s] {self.source} --> {self.target}"
        else:
            return f"[{self.trip_id}] [{self.source_arrival_time}] {self.source}  -> {self.target} [{self.target_departure_time}]"

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.trip_id))

    def __eq__(self, other) -> bool:
        return self.source == other.source and self.target == other.target and self.trip_id == other.trip_id
