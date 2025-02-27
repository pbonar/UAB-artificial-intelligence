# SearchAlgorithm.py

# This file contains all the required routines to make an A* search algorithm.
#
__author__ = '1759684'
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
        total_cost = 0  # Initialize total cost for the entire path
        for i in range(len(path.route) - 1):
            station1 = path.route[i]
            station2 = path.route[i + 1]
            if station1 in map.connections and station2 in map.connections[station1]:
                if type_preference == 0:
                    # Cost is the number of stations traveled (adjacency)
                    total_cost += 1
                elif type_preference == 1:
                    # Cost is the distance between stations (in meters)
                    time_cost = map.connections[station1][station2]
                    total_cost += time_cost
                elif type_preference == 2:
                    velocity = map.stations[station1]['velocity']
                    distance_cost = map.connections[station1][station2] * velocity
                    total_cost += distance_cost
                elif type_preference == 3:
                    # Cost is the number of transfers (penalty for changing lines)
                    if map.stations[station1]['line'] != map.stations[station2]['line']:
                        total_cost += 1  # Penalty for transfer
        path.g = total_cost  # Update the total cost of the path
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
    pass


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
    # Get the coordinates of the destination station
    pass


def update_f(expand_paths):
    """
      Update the f of a path
      Format of the parameter is:
         Args:
             expand_paths (LIST of Path Class): Expanded paths
         Returns:
             expand_paths (LIST of Path Class): Expanded paths with updated costs
    """
    pass


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
    pass


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
    pass


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
    pass


def Astar_improved(origin_coord, destination_coord, map):
    """
     A* Search algorithm
     Format of the parameter is:
        Args:
            origin_coord (list): Two REAL values, which refer to the coordinates of the starting position
            destination_coord (list): Two REAL values, which refer to the coordinates of the final position
            map (object of Map class): All the map information

        Returns:
            list_of_path[0] (Path Class): The route that goes from origin_coord to destination_coord
    """
    pass

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
    print(breadth_first_search( 1, 10, map).route)
    print(depth_first_search( 1, 10, map).route)

    print(distance_to_stations([0, 0], map))
