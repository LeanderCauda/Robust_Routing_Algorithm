class Node:
    def __init__(
        self,
        stop_id: str,
        stop_name: str,
        stop_lon: float,
        stop_lat: float,
        parent_stop_id: str
    ):
        self.stop_id = stop_id
        self.stop_name = stop_name
        self.stop_lon = stop_lon
        self.stop_lat = stop_lat
        self.parent_stop_id = parent_stop_id

    def id(self) -> str:
        return self.stop_id

    def __repr__(self) -> str:
        return f"{self.stop_name} ({self.stop_id})"

    def __hash__(self) -> int:
        return hash(self.stop_id)

    def __eq__(self, other) -> bool:
        return (
            self.stop_id == other.stop_id 
            and self.stop_name == other.stop_name 
            and self.stop_lon == other.stop_lon 
            and self.stop_lat == other.stop_lat
        )
