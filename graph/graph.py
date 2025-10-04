from graph.edge import Edge
from graph.node import Node
from datetime import datetime, time
from collections import defaultdict
from pyspark.sql import DataFrame
from zoneinfo import ZoneInfo


class MultiDiGraph:
    def __init__(self):
        self.nodes = {}
        self.adjacency = defaultdict(set)

    def add_node(self, node: Node):
        if node.id() in self.nodes:
            if node != self.nodes[node.id()]:
                raise ValueError("Trying to add a node with existing id but different attributes")
            else: return
        self.nodes[node.id()] = node

    def add_edge(self, edge: Edge):
        self.adjacency[edge.source.id()].add((edge.target.id(), edge))

    def remove_edge(self, edge: Edge) -> bool:
        if not (edge.target.id(), edge) in self.adjacency[edge.source.id()]:
            return False
        self.adjacency[edge.source.id()].remove((edge.target.id(), edge))
        return True

    def __getitem__(self, node_id: str) -> Node:
        if not node_id in self.nodes:
            raise ValueError(f"Node with id {node_id} does not exist in the graph")
        return self.nodes[node_id]

    def neighbors(self, node_id: str) -> set[tuple[str, Edge]]:
        return self.adjacency[node_id]
        
    @staticmethod
    def from_spark_dataframe(dataframe: DataFrame, arrival_datetime: datetime) -> 'MultiDiGraph':
        graph = MultiDiGraph()
        for row in dataframe.collect():
            source_node = Node(
                row.source_stop_id,
                row.source_stop_name,
                row.source_stop_lon,
                row.source_stop_lat,
                row.source_parent_stop_id
            )

            target_node = Node(
                row.target_stop_id,
                row.target_stop_name,
                row.target_stop_lon,
                row.target_stop_lat,
                row.target_parent_stop_id
            )

            if row.trip_id == "walking":
                source_arrival_time = int(row.source_arrival_time)
                target_departure_time = 0
            else:
                source_arrival_time = datetime.combine(
                    arrival_datetime, 
                    datetime.strptime(row.source_arrival_time, "%H:%M:%S").time()
                ).replace(tzinfo=ZoneInfo("Europe/Zurich"))
                target_departure_time = datetime.combine(
                    arrival_datetime, 
                    datetime.strptime(row.target_departure_time, "%H:%M:%S").time()
                ).replace(tzinfo=ZoneInfo("Europe/Zurich"))

            graph.add_node(source_node)
            graph.add_node(target_node)
            graph.add_edge(Edge(
                source_node,
                target_node,
                row.trip_id,
                row.trip_description,
                row.transfer_time,
                source_arrival_time,
                target_departure_time
            ))
        
        return graph
