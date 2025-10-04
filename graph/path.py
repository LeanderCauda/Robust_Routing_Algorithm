# +
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import copy
from graph.edge import Edge
from graph.node import Node
from graph.trip import Trip
from datetime import datetime, timedelta

MAPBOX_TOKEN = "pk.eyJ1IjoiamVhbi1zaWZmZXJ0IiwiYSI6ImNtOXdrejNneDB0ZWgya3NlMzNydmFrYnUifQ.JlZfSh_BG8YL-P_A1K2jfw"
px.set_mapbox_access_token(MAPBOX_TOKEN)


# -

class Path:
    def __init__(
        self, 
        trips: list[Trip], 
        expected_arrival_datetime: datetime, 
        dijkstra_cost: float, 
        walked_distance: float, 
        walking_speed: float,
        maximum_walking_distance: int,
        is_compressed: bool = False,
        confidence: float = None
    ):
        self.trips = trips
        self.dijkstra_cost = dijkstra_cost
        self.expected_arrival_datetime = expected_arrival_datetime
        self.walked_distance = walked_distance
        self.walking_speed = walking_speed
        self.maximum_walking_distance = maximum_walking_distance
        self.is_compressed = is_compressed
        self.confidence = confidence

        self.departure_time = self.trips[0].source_departure_time
        self.arrival_time = self.trips[-1].target_arrival_time
        
        self.transfers_count = self.__count_transfers()

    def __count_transfers(self) -> int:
        transfer = 0
        last_trip_id = None
        for trip in self.trips:
            if trip.trip_id != last_trip_id:
                if last_trip_id != None:
                    transfer += 1
                last_trip_id = trip.trip_id
        return transfer

    def compress(self) -> 'Path':
        i = 0
        trips = []
        while i < len(self.trips):
            if self.trips[i].is_walking():
                trips.append(copy.deepcopy(self.trips[i]))
            else:
                j = i + 1
                while j < len(self.trips) and self.trips[i].trip_id == self.trips[j].trip_id:
                    j += 1
                j -= 1

                trips.append(Trip(
                    source=self.trips[i].source,
                    target=self.trips[j].target,
                    trip_id=self.trips[i].trip_id,
                    trip_description=self.trips[i].trip_description,
                    source_departure_time=self.trips[i].source_departure_time,
                    target_arrival_time=self.trips[j].target_arrival_time,
                ))
                i = j
            i += 1
        
        return Path(
            trips,
            expected_arrival_datetime=self.expected_arrival_datetime, 
            dijkstra_cost=self.dijkstra_cost,
            walked_distance=self.walked_distance,
            walking_speed=self.walking_speed,
            maximum_walking_distance=self.maximum_walking_distance,
            is_compressed=True
        )   

    def cost(self) -> tuple:
        return (self.dijkstra_cost, self.transfers_count, self.walked_distance)

    def __repr__(self) -> str:
        start_stop_name = self.trips[0].source.stop_name
        end_stop_name = self.trips[-1].target.stop_name
        confidence_string = f"{self.confidence:.4f}" if self.confidence else "Not computed"
        lines = [
            "[*]" + "-" * 64 + "[*]",
            f"[*] {start_stop_name} --> {end_stop_name}",
            f"[*] Departure time  : {self.departure_time.time()}",
            f"[*] Arrival time    : {self.arrival_time.time()} (desired {self.expected_arrival_datetime.time()})",
            f"[*] Walked distance : {self.walked_distance:.2f}m / {self.maximum_walking_distance}m",
            f"[*] Transfers       : {self.transfers_count}",
            f"[*] Cost            : {self.dijkstra_cost}",
            f"[*] Cost computed   : {(self.arrival_time - self.departure_time).total_seconds()}",
            f"[*] Confidence      : {confidence_string}",
            "[*]" + "-" * 64 + "[*]"
        ]
        
        last_trip_id = None
        for trip in self.trips:
            source_stop = trip.source
            target_stop = trip.target
            transfer_string = "[TRANSFER]" if (last_trip_id != trip.trip_id and last_trip_id) else ""
            if trip.trip_id == "walking":
                walking_time = (trip.target_arrival_time - trip.source_departure_time).total_seconds()
                walking_distance = walking_time * self.walking_speed
                lines.append(f"[*] {transfer_string} {trip.trip_description} (time = {walking_time:.2f}s | distance = {walking_distance:.2f}m)")
                lines.append(f"[*] [{trip.source_departure_time.time()}] ({source_stop.stop_id:9}) {source_stop.stop_name}")
                lines.append(f"[*] [{trip.target_arrival_time.time()}] ({target_stop.stop_id:9}) {target_stop.stop_name}")
            else:
                lines.append(f"[*] {transfer_string} {trip.trip_description} ({trip.trip_id})")
                lines.append(f"[*] [{trip.source_departure_time.time()}] ({source_stop.stop_id:9}) {source_stop.stop_name}")
                lines.append(f"[*] [{trip.target_arrival_time.time()}] ({target_stop.stop_id:9}) {target_stop.stop_name}")
            lines.append("[*]" + "-" * 64 + "[*]")
            last_trip_id = trip.trip_id
        return "\n".join(lines)
        
    def __hash__(self) -> int:
        return hash(tuple(self.trips))

    def __eq__(self, other) -> bool:
        s, o = self.trips, other.trips
        if len(s) != len(o): return False
        return all([s_e == o_e for s_e, o_e in zip(s, o)])

    @staticmethod
    def from_dijkstra_output(
        dijkstra_output: tuple,
        arrival_datetime: datetime,
        maximum_walking_distance: int,
        walking_speed: float
    ) -> 'Path':
        dijkstra_cost, path = dijkstra_output
        walked_distance = maximum_walking_distance - path[-1]['walking_budget']
        departure_time = path[-1]['latest_departure']
        current_time = path[-1]['latest_departure']

        trips = []
        for edge in [p['edge'] for p in path[1:]][::-1]:
            if edge.is_walking():
                trips.append(Trip(
                    source=edge.target,
                    target=edge.source,
                    trip_id=edge.trip_id,
                    trip_description=edge.trip_description,
                    source_departure_time=current_time,
                    target_arrival_time=current_time + timedelta(seconds=edge.source_arrival_time)
                ))
            else:
                trips.append(Trip(
                    source=edge.target,
                    target=edge.source,
                    trip_id=edge.trip_id,
                    trip_description=edge.trip_description,
                    source_departure_time=edge.target_departure_time,
                    target_arrival_time=edge.source_arrival_time
                ))

                current_time = edge.source_arrival_time
        
        return Path(
            trips=trips,
            expected_arrival_datetime=arrival_datetime, 
            dijkstra_cost=dijkstra_cost,
            walked_distance=maximum_walking_distance - path[-1]['walking_budget'],
            walking_speed=walking_speed,
            maximum_walking_distance=maximum_walking_distance
        )
