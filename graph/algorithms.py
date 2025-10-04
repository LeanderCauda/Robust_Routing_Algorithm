import heapq
import itertools
import sys
from collections import defaultdict
from graph.edge import Edge
from graph.path import Path
from graph.graph import MultiDiGraph
from datetime import datetime, timedelta


def dijkstra_cost_state(
    edge: Edge, 
    current_cost: int, 
    current_state: dict, 
    walking_speed: float,
    default_transfer_time: int
) -> tuple[float, dict]:
    if edge.is_walking():
        walking_time = edge.source_arrival_time
        walking_distance = walking_time * walking_speed
        
        if current_state['walking_budget'] < walking_distance:
            raise ValueError("Walking budget exceeded")

        edge_cost = max(edge.transfer_time, walking_time)
        cost = current_cost + edge_cost
        state = {
            'stop_id': edge.target.id(),
            'latest_departure': current_state['latest_departure'] - timedelta(seconds=edge_cost),
            'walking_budget': current_state['walking_budget'] - walking_distance,
            'previous_trip_id': 'walking',
            'edge': edge
        }
    else:
        if edge.source_arrival_time > current_state['latest_departure']:
            raise ValueError("Trip arrival time exceed deadline")
        
        if edge.trip_id != current_state['previous_trip_id']:
            if current_state['previous_trip_id'] != "walking" and edge.source_arrival_time > current_state['latest_departure'] - timedelta(seconds=default_transfer_time):
                raise ValueError("Trip arrival time exceed deadline (no time for transfer)")
            transfer_time = (current_state['latest_departure'] - edge.source_arrival_time).total_seconds()
        else:
            transfer_time = 0
        travel_time = (edge.source_arrival_time - edge.target_departure_time).total_seconds()
        cost = current_cost + transfer_time + travel_time
        state = {
            'stop_id': edge.target.id(),
            'latest_departure': edge.target_departure_time,
            'walking_budget': current_state['walking_budget'],
            'previous_trip_id': edge.trip_id,
            'edge': edge
        }
    
    return cost, state


def dijkstra(
    graph: MultiDiGraph, 
    source_id: str, 
    target_id: str, 
    arrival_datetime: datetime,
    maximum_walking_distance: int,
    walking_speed: float,
    default_transfer_time: int,
    previous_trip_id: str = None
) -> tuple[float, list[dict]]:
    heap    = []
    visited = defaultdict(bool)
    costs   = defaultdict(lambda: float('inf'))
    counter = itertools.count() # tie breaking in heap
    
    costs[source_id] = 0
    heapq.heappush(heap, (0, next(counter), [{
        'stop_id': source_id,
        'latest_departure': arrival_datetime,
        'walking_budget': maximum_walking_distance,
        'previous_trip_id': previous_trip_id,
        'edge': None
    }]))

    while heap:
        cost, _, states = heapq.heappop(heap)
        current_state   = states[-1]
        current_node_id = current_state['stop_id']
        
        visited[current_node_id] = True

        if current_node_id == target_id:
            return cost, states
        
        if costs[current_node_id] < cost: continue # Don't process stale pairs

        for edge_target_id, edge in graph.neighbors(current_node_id):
            if visited[edge_target_id]:
                continue
            try:
                new_cost, new_state = dijkstra_cost_state(
                    edge, 
                    cost, 
                    current_state, 
                    walking_speed,
                    default_transfer_time
                )
    
                if new_cost < costs[edge_target_id]:
                    costs[edge_target_id] = new_cost
                    heapq.heappush(heap, (new_cost, next(counter), states + [new_state]))
            except ValueError as err:
                pass
    return float('inf'), []


def are_same_paths(path_1: list, path_2: list) -> bool:
    filtered_path_1 = tuple(p['edge'] for p in path_1[1:])
    filtered_path_2 = tuple(p['edge'] for p in path_2[1:])
    
    if len(filtered_path_1) == 0 and len(filtered_path_2) == 0:
        return True
    return filtered_path_1 == filtered_path_2


def yens_k_shortest_paths(
    graph: MultiDiGraph, 
    source_id: str, 
    target_id: str, 
    K: int,
    arrival_datetime: datetime,
    maximum_walking_distance: int,
    walking_speed: float,
    default_transfer_time: int,
    previous_trip_id: str = None
) -> list[tuple[float, list[dict]]]:
    assert K >= 1
    cost, path = dijkstra(
        graph, 
        source_id, 
        target_id, 
        arrival_datetime, 
        maximum_walking_distance, 
        walking_speed, 
        default_transfer_time, 
        previous_trip_id
    )

    if cost == float('inf'):
        raise ValueError("Unreachable stop")

    paths      = [(cost, path)]
    counter    = itertools.count() # tie breaking in heap
    candidates = []
    
    for k in range(K - 1):
        last_path = paths[-1][1]
        for i in range(len(last_path) - 1):
            spur_node = last_path[i]
            root_path = last_path[:i+1]

            removed_edges = []
            for found_cost, found_path in paths:
                if are_same_paths(found_path[:i+1], root_path) and len(found_path) > i + 1:
                    edge_to_remove = found_path[i+1]['edge']
                    if graph.remove_edge(edge_to_remove):
                        removed_edges.append(edge_to_remove)

            spur_cost, spur_path = dijkstra(
                graph,
                source_id=spur_node['stop_id'],
                target_id=target_id,
                arrival_datetime=spur_node['latest_departure'],
                maximum_walking_distance=spur_node['walking_budget'],
                walking_speed=walking_speed,
                default_transfer_time=default_transfer_time,
                previous_trip_id=spur_node['previous_trip_id']
            )
            
            for edge in removed_edges:
                graph.add_edge(edge)

            if spur_cost != float('inf'):
                total_path = root_path + spur_path[1:]
                total_cost = (arrival_datetime - total_path[-1]['latest_departure']).total_seconds()
                candidate = (total_cost, next(counter), total_path)

                flag = True
                for candidate_path in candidates:
                    if are_same_paths(candidate_path[2], total_path):
                        flag = False

                if flag:
                    heapq.heappush(candidates, candidate)
        
        if not candidates:
            break

        total_cost, _, total_path = heapq.heappop(candidates)
        paths.append((total_cost, total_path))
    return paths
