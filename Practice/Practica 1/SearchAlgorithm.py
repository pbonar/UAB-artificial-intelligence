# SearchAlgorithm.py

# This file contains all the required routines to make an A* search algorithm.
#
__author__ = '1759684'

import heapq

# _________________________________________________________________________________________
# Intel.ligencia Artificial
# Curs 2023 - 2024
# Universitat Autonoma de Barcelona
# _______________________________________________________________________________________

from SubwayMap import *
from utils import *
import os
import math
import copy


def expand(path, map):
    """
    Expands a SINGLE station and returns a list of new Path objects.

    Args:
        path (Path): The current path to expand.
        map (Map): The transit system containing station connections.

    Returns:
        list: A list of new Path objects representing possible routes.
    """
    path_list = []
    if path.last not in map.connections:
        return path_list

    for station, cost in map.connections[path.last].items():
        new_path = Path(path.route + [station])
        new_path.update_g(path.g + cost)
        new_path.update_f()
        path_list.append(new_path)

    return path_list


def remove_cycles(path_list):
    """
     It removes from path_list the set of paths that include some cycles in their path.
     Format of the parameter is:
        Args:
            path_list (LIST of Path Class): Expanded paths
        Returns:
            path_list (list): Expanded paths without cycles.
    """

    return [path for path in path_list if len(path.route) == len(set(path.route))]


def insert_depth_first_search(expand_paths, list_of_path):
    """
     expand_paths is inserted to the list_of_path according to DEPTH FIRST SEARCH algorithm
     Format of the parameter is:
        Args:
            expand_paths (LIST of Path Class): Expanded paths
            list_of_path (LIST of Path Class): The paths to be visited
        Returns:
            list_of_path (LIST of Path Class): List of Paths where Expanded Path is inserted
    """
    return expand_paths + list_of_path


def depth_first_search(origin_id, destination_id, map):
    """
     Depth First Search algorithm
     Format of the parameter is:
        Args:
            origin_id (int): Starting station id
            destination_id (int): Final station id
            map (object of Map class): All the map information
        Returns:
            list_of_path[0] (Path Class): the route that goes from origin_id to destination_id
    """
    stack = [Path(origin_id)]
    while stack:
        path = stack.pop(0)
        if path.last == destination_id:
            return path
        expanded_paths = expand(path, map)
        expanded_paths = remove_cycles(expanded_paths)
        stack = insert_depth_first_search(expanded_paths, stack)
    return None


def insert_breadth_first_search(expand_paths, list_of_path):
    """
        expand_paths is inserted to the list_of_path according to BREADTH FIRST SEARCH algorithm
        Format of the parameter is:
           Args:
               expand_paths (LIST of Path Class): Expanded paths
               list_of_path (LIST of Path Class): The paths to be visited
           Returns:
               list_of_path (LIST of Path Class): List of Paths where Expanded Path is inserted
    """
    return list_of_path + expand_paths


def breadth_first_search(origin_id, destination_id, map):
    """
     Breadth First Search algorithm
     Format of the parameter is:
        Args:
            origin_id (int): Starting station id
            destination_id (int): Final station id
            map (object of Map class): All the map information
        Returns:
            list_of_path[0] (Path Class): The route that goes from origin_id to destination_id
    """
    queue = [Path(origin_id)]
    while queue:
        path = queue.pop(0)
        if path.last == destination_id:
            return path
        expanded_paths = expand(path, map)
        expanded_paths = remove_cycles(expanded_paths)
        queue = insert_breadth_first_search(expanded_paths, queue)
    return None


def calculate_cost(expand_paths, map, type_preference=0):
    """
    Calculate the cost according to type preference
    Format of the parameter is:
        Args:
            expand_paths (LIST of Paths Class): Expanded paths
            map (object of Map class): All the map information
            type_preference: INTEGER Value to indicate the preference selected:
                            0 - Adjacency
                            1 - minimum Time
                            2 - minimum Distance
                            3 - minimum Transfers
        Returns:
            expand_paths (LIST of Paths): Expanded path with updated cost
    """
    for path in expand_paths:
        total_cost = 0
        for i in range(len(path.route) - 1):
            station1 = path.route[i]
            station2 = path.route[i + 1]
            if station1 in map.connections and station2 in map.connections[station1]:
                if type_preference == 0:
                    total_cost += 1
                elif type_preference == 1:
                    time_cost = map.connections[station1][station2]
                    total_cost += time_cost
                elif type_preference == 2:
                    velocity = map.stations[station1]['velocity']
                    distance_cost = map.connections[station1][station2] * velocity
                    total_cost += distance_cost
                elif type_preference == 3:
                    if map.stations[station1]['line'] != map.stations[station2]['line']:
                        total_cost += 1
        path.g = total_cost
    return expand_paths



def insert_cost(expand_paths, list_of_path):
    """
        expand_paths is inserted to the list_of_path according to COST VALUE
        Format of the parameter is:
           Args:
               expand_paths (LIST of Path Class): Expanded paths
               list_of_path (LIST of Path Class): The paths to be visited
           Returns:
               list_of_path (LIST of Path Class): List of Paths where expanded_path is inserted according to cost
    """
    return sorted(list_of_path + expand_paths, key=lambda path: path.g)


def uniform_cost_search(origin_id, destination_id, map, type_preference=0):
    """
    Uniform Cost Search algorithm
    Format of the parameter is:
        Args:
            origin_id (int): Starting station id
            destination_id (int): Final station id
            map (object of Map class): All the map information
            type_preference: INTEGER Value to indicate the preference selected:
                            0 - Adjacency
                            1 - minimum Time
                            2 - minimum Distance
                            3 - minimum Transfers
        Returns:
            Path: The cheapest route that goes from origin_id to destination_id
    """
    priority_queue = [Path([origin_id])]
    visited_stations_cost = {}
    best_path = None
    best_cost = float('inf')

    while priority_queue:
        next_queue = []

        print("Current priority queue:", [f"{p.route} {p.g}" for p in priority_queue])

        for path in priority_queue:
            if path.last == destination_id:
                if path.g < best_cost:
                    best_cost = path.g
                    best_path = path
                continue

            if path.last in visited_stations_cost and path.g >= visited_stations_cost[path.last]:
                continue

            visited_stations_cost[path.last] = path.g

            expanded_paths = expand(path, map)

            expanded_paths = remove_cycles(expanded_paths)

            calculate_cost(expanded_paths, map, type_preference)

            next_queue.extend(expanded_paths)

        priority_queue = sorted(next_queue, key=lambda p: p.g)

    # print("Picked: ", best_path.route)
    return best_path


def calculate_heuristics(expand_paths, map, destination_id, type_preference=0):
    """
    Calculate and UPDATE the heuristics of a path according to type preference.
    The heuristic is an estimate of the cost from the current station to the destination.

    Args:
        expand_paths (LIST of Path Class): Expanded paths
        map (object of Map class): All the map information
        destination_id (int): Final station id
        type_preference: INTEGER Value to indicate the preference selected:
                        0 - Adjacency
                        1 - minimum Time
                        2 - minimum Distance
                        3 - minimum Transfers
    Returns:
        expand_paths (LIST of Path Class): Expanded paths with updated heuristics
    """
    destination_coords = (map.stations[destination_id]['x'], map.stations[destination_id]['y'])

    for path in expand_paths:
        last_station = path.last
        last_coords = (map.stations[last_station]['x'], map.stations[last_station]['y'])

        distance = euclidean_dist(last_coords, destination_coords)
        # current_line = map.stations[last_station]['line']
        max_velocity = max(map.velocity.values()) if map.velocity else 10

        if type_preference == 0:
            path.h = 1 if path.last != destination_id else 0
        if type_preference == 1:
            path.h = distance / max_velocity
        elif type_preference == 2:
            path.h = distance
        elif type_preference == 3:
            print("Last: ", map.stations[last_station]['line'])
            print("Dest: ", map.stations[destination_id]['line'])
            current_line = map.stations[last_station]['line']
            destination_line = map.stations[destination_id]['line']
            if current_line == destination_line:
                path.h = 0
            else:
                path.h = 1

    return expand_paths


def update_f(expand_paths):
    """
      Update the f of a path
      Format of the parameter is:
         Args:
             expand_paths (LIST of Path Class): Expanded paths
         Returns:
             expand_paths (LIST of Path Class): Expanded paths with updated costs
    """
    for path in expand_paths:
        path.update_f()

    return expand_paths


def remove_redundant_paths(expand_paths, list_of_path, visited_stations_cost):
    """
      It removes the Redundant Paths. They are not optimal solution!
      If a station is visited and have a lower g-cost at this moment, we should remove this path.
      Format of the parameter is:
         Args:
             expand_paths (LIST of Path Class): Expanded paths
             list_of_path (LIST of Path Class): All the paths to be expanded
             visited_stations_cost (dict): All visited stations cost
         Returns:
             new_paths (LIST of Path Class): Expanded paths without redundant paths
             list_of_path (LIST of Path Class): list_of_path without redundant paths
             visited_stations_cost (dict): Updated visited stations cost
    """
    new_paths = []
    for path in expand_paths:
        if path.last not in visited_stations_cost or path.g <= visited_stations_cost[path.last]:
            visited_stations_cost[path.last] = path.g
            new_paths.append(path)

    filtered_list_of_path = [
        path for path in list_of_path
        if path.last not in visited_stations_cost or path.g <= visited_stations_cost[path.last]
    ]

    return new_paths, filtered_list_of_path, visited_stations_cost


def insert_cost_f(expand_paths, list_of_path):
    """
        expand_paths is inserted to the list_of_path according to f VALUE
        Format of the parameter is:
           Args:
               expand_paths (LIST of Path Class): Expanded paths
               list_of_path (LIST of Path Class): The paths to be visited
           Returns:
               list_of_path (LIST of Path Class): List of Paths where expanded_path is inserted according to f
    """
    list_of_path.extend(expand_paths)
    list_of_path.sort(key=lambda path: path.f)

    return list_of_path


def distance_to_stations(coord, map):
    """
        From coordinates, it computes the distance to all stations in map.
        Format of the parameter is:
        Args:
            coord (list): Two REAL values, which refer to the coordinates of a point in the city.
            map (object of Map class): All the map information
        Returns:
            (dict): Dictionary containing as keys, all the Indexes of all the stations in the map, and as values, the
            distance between each station and the coord point
    """
    distances = {}
    user_x, user_y = coord

    for station_id, station_info in map.stations.items():
        station_x = station_info['x']
        station_y = station_info['y']
        dist = euclidean_dist((user_x, user_y), (station_x, station_y))
        distances[station_id] = dist

    sorted_distances = dict(sorted(distances.items(), key=lambda x: (x[1], x[0])))
    return sorted_distances

def Astar(origin_id, destination_id, map, type_preference=0):
    """
     A* Search algorithm
     Format of the parameter is:
        Args:
            origin_id (int): Starting station id
            destination_id (int): Final station id
            map (object of Map class): All the map information
            type_preference: INTEGER Value to indicate the preference selected:
                            0 - Adjacency
                            1 - minimum Time
                            2 - minimum Distance
                            3 - minimum Transfers
        Returns:
            list_of_path[0] (Path Class): The route that goes from origin_id to destination_id
    """
    priority_queue = [Path([origin_id])]
    visited_stations_cost = {}

    while priority_queue:
        path = priority_queue.pop(0)
        if path.last == destination_id:
            return path
        print(path)
        expanded_paths = expand(path, map)
        calculate_cost(expanded_paths, map, type_preference)
        calculate_heuristics(expanded_paths, map, destination_id, type_preference)
        update_f(expanded_paths)
        expanded_paths, priority_queue, visited_stations_cost = remove_redundant_paths(expanded_paths, priority_queue,
                                                                                       visited_stations_cost)
        priority_queue = insert_cost_f(expanded_paths, priority_queue)
    return None

def Astar_improved(origin_coord, destination_coord, map):
    """
    A* Search algorithm that finds the optimal time path given origin and destination coordinates.

    Args:
        origin_coord (list): Two REAL values representing the starting position coordinates.
        destination_coord (list): Two REAL values representing the final position coordinates.
        map (dict): A dictionary containing station IDs as keys and station coordinates as values.

    Returns:
        Path (Path Class): The optimal route that goes from origin_coord to destination_coord.
    """
    walking_speed = 5

    # Get the distance to stations for both the origin and destination
    start_dict = distance_to_stations(origin_coord, map)
    end_dict = distance_to_stations(destination_coord, map)

    priority_queue = []
    visited_stations_cost = {}

    # Initialize the priority queue with the starting points
    for station_id, distance in start_dict.items():
        time_to_station = distance / walking_speed
        heuristic_to_destination = end_dict[station_id] / walking_speed
        path = Path([0, station_id])
        path.g = time_to_station
        path.h = heuristic_to_destination
        path.update_f()
        priority_queue.append(path)

    # Direct walk from origin to destination
    direct_walk_time = distance_to_stations(origin_coord, {0: destination_coord})[0] / walking_speed
    direct_path = Path([0, -1])
    direct_path.g = direct_walk_time
    direct_path.h = 0
    direct_path.f = direct_path.g + direct_path.h
    priority_queue.append(direct_path)

    # Sort the queue based on f values
    priority_queue.sort(key=lambda x: x.f)

    while priority_queue:
        path = priority_queue.pop(0)  # Pop the path with the lowest f value

        if path.last == -1:  # Reached destination
            return path

        if path.last in visited_stations_cost and visited_stations_cost[path.last] <= path.g:
            continue
        visited_stations_cost[path.last] = path.g

        # Expand paths from stations
        if path.last != 0:
            expanded_paths = expand(path, map)
            calculate_cost(expanded_paths, map, 1)  # Minimize time
            calculate_heuristics(expanded_paths, map, None, 1)
            update_f(expanded_paths)

            # Remove redundant paths (those that have already been visited with a lower cost)
            expanded_paths, priority_queue, visited_stations_cost = remove_redundant_paths(
                expanded_paths, priority_queue, visited_stations_cost
            )

            # Insert expanded paths into priority_queue and sort it
            priority_queue.extend(expanded_paths)
            priority_queue.sort(key=lambda x: x.f)  # Keep the queue sorted by f

            # Option to walk directly to the destination
            if path.last in end_dict:
                walk_to_dest_time = end_dict[path.last] / walking_speed
                final_path = Path(path.route + [-1])
                final_path.g = path.g + walk_to_dest_time
                final_path.h = 0
                final_path.f = final_path.g + final_path.h
                priority_queue.append(final_path)
                priority_queue.sort(key=lambda x: x.f)  # Keep the queue sorted by f

    return None


# Just for testing
if __name__ == "__main__":
    ROOT_FOLDER = 'CityInformation/Lyon_smallCity/'
    subway_map = read_station_information(os.path.join(ROOT_FOLDER, 'Stations.txt'))
    connections = read_cost_table(os.path.join(ROOT_FOLDER, 'Time.txt'))
    subway_map.add_connection(connections)

    info_velocity_clean = read_information(os.path.join(ROOT_FOLDER, 'InfoVelocity.txt'))
    subway_map.add_velocity(info_velocity_clean)

    map = subway_map

    # for path in ...:
    #     print(path.route)
    # print(breadth_first_search( 1, 10, map).route)
    # print(depth_first_search( 1, 10, map).route)
    #
    # print(distance_to_stations([0, 0], map))
    # Astar_improved([0,0], [200,200], map)

    print("hola: ", Astar(4,14, map, 1).route, Astar(4,14, map, 1).g)

# if __name__ == "__main__":
#     ROOT_FOLDER = 'CityInformation/Lyon_smallCity/'
#     subway_map = read_station_information(os.path.join(ROOT_FOLDER, 'Stations.txt'))
#     connections = read_cost_table(os.path.join(ROOT_FOLDER, 'Time.txt'))
#     subway_map.add_connection(connections)
#
#     info_velocity_clean = read_information(os.path.join(ROOT_FOLDER, 'InfoVelocity.txt'))
#     subway_map.add_velocity(info_velocity_clean)
#
#     map = subway_map
#     destination_id = 10  # Set this to a station ID that will be your destination
#
#     # Get all stations and calculate heuristics for distance (Euclidean distance)
#     best_heuristic = float('inf')
#     best_station = None
#
#     # Iterate over all stations to calculate heuristics
#     for station_id in map.stations:
#         path = Path([station_id])
#         expand_paths = [path]  # Expand the initial path
#         expanded_paths = calculate_heuristics(expand_paths, map, destination_id, type_preference=2)  # 2 is for distance
#         for expanded_path in expanded_paths:
#             if expanded_path.h < best_heuristic:
#                 best_heuristic = expanded_path.h
#                 best_station = station_id
#
#     # Output the result
#     print(f"The best value of heuristics for distance is {best_heuristic} for station {best_station}")
