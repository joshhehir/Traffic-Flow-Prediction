from data.scats import ScatsData
import math
from math import radians, cos, sin, asin, sqrt
import json
import numpy as np
from keras.models import load_model

SCATS_DATA = ScatsData()


def angle_of_vectors(a, b, c, d):
    dotProduct = a * c + b * d
    # for three dimensional simply add dotProduct = a*c + b*d  + e*f
    modOfVector1 = math.sqrt(a * a + b * b) * math.sqrt(c * c + d * d)
    # for three dimensional simply add modOfVector = math.sqrt( a*a + b*b + e*e)*math.sqrt(c*c + d*d +f*f)
    angle = dotProduct / modOfVector1
    angleInDegree = math.degrees(math.acos(angle))
    # print("θ =",angleInDegree,"°")
    return angleInDegree


def get_streets_from_name(name):
    words = name.upper().split(' OF ')
    streetB = words[1]
    streetA = words[0].split()
    streetA = ' '.join(streetA[:len(streetA) - 1])
    return [streetA, streetB]


def get_cardinality_from_name(name):
    words = name.upper().split(' OF ')
    cardinality = words[0].split()
    cardinality = cardinality[len(cardinality) - 1:][0]
    if cardinality == "N":
        return 0
    elif cardinality == "NE":
        return 1
    elif cardinality == "E":
        return 2
    elif cardinality == "SE":
        return 3
    elif cardinality == "S":
        return 4
    elif cardinality == "SW":
        return 5
    elif cardinality == "W":
        return 6
    elif cardinality == "NW":
        return 7
    else:
        return None


def convert_cardinality_to_vector(cardinality):
    if cardinality == 0:
        return 0, 1
    elif cardinality == 1:
        return 1, 1
    elif cardinality == 2:
        return 1, 0
    elif cardinality == 3:
        return 1, -1
    elif cardinality == 4:
        return 0, -1
    elif cardinality == 5:
        return -1, -1
    elif cardinality == 6:
        return -1, 0
    elif cardinality == 7:
        return -1, 1


def get_inverse_cardinality(cardinality):
    reverse = cardinality + 4
    if reverse >= 8:
        reverse = reverse - 8
    return reverse


def distance(vector_a, vector_b):
     
    # The math module contains a function named
    # radians which converts from degrees to radians.
    lon1 = radians(vector_a[0])
    lon2 = radians(vector_b[0])
    lat1 = radians(vector_a[1])
    lat2 = radians(vector_b[1])
      
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
 
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371
      
    # calculate the result
    return(c * r)


def direction(vector_a, vector_b):
    return [vector_a[0] - vector_b[0], vector_a[1] - vector_b[1]]


class Node(object):
    def __init__(self, scats_number, coordinates):
        self.scats_number = scats_number
        self.coordinates = coordinates
        self.incoming_connections = []
        self.outgoing_connections = []

    def add_incoming_connection(self, connection):
        self.incoming_connections.append(connection)

    def find_outgoing_connections(self, graph):
        for connection in self.incoming_connections:
            connection = self.get_respective_outgoing_connection(connection, graph)
            if connection != None:
                self.outgoing_connections.append(connection)

    def get_respective_outgoing_connection(self, connection, graph):
        """1. get connections of connected nodes
        2. get nodes in direction of cardinality
        3. get closest node """

        valid_connections = []
        for node in graph.nodes:
            for external_connection in node.incoming_connections:
                if external_connection.streets[0] in connection.streets or external_connection.streets[
                    1] in connection.streets:
                    if not (external_connection.streets[0] in connection.streets and external_connection.streets[
                        1] in connection.streets):
                        vector_a = convert_cardinality_to_vector(connection.cardinality)
                        vector_b = tuple(
                            map(lambda i, j: i - j, external_connection.node.coordinates, connection.node.coordinates))
                        if angle_of_vectors(vector_a[1], vector_a[0], vector_b[0], vector_b[1]) < 45:
                            valid_connections.append(external_connection)

        best_connection = None
        for external_connection in valid_connections:
            if best_connection == None:
                best_connection = external_connection
            elif (distance(connection.node.coordinates, external_connection.node.coordinates) < distance(
                    connection.node.coordinates, best_connection.node.coordinates)):
                best_connection = external_connection
            elif (distance(connection.node.coordinates, external_connection.node.coordinates) == distance(
                    connection.node.coordinates, best_connection.node.coordinates)):
                if abs(get_inverse_cardinality(connection.cardinality) - external_connection.cardinality) < abs(
                        get_inverse_cardinality(connection.cardinality) - best_connection.cardinality):
                    best_connection = external_connection
        return best_connection

    def add_outgoing_connection(self, connection):
        if connection in self.outgoing_connections:
            return
        self.outgoing_connections.append(connection)

    def get_streets_in_connections(self):
        streets = []
        for connection in self.incoming_connections:
            for street in connection.streets:
                if street not in streets:
                    streets.append(street)
        return streets


class Connection(object):

    def __init__(self, name, node):
        self.node = node
        self.streets = get_streets_from_name(name)
        self.cardinality = get_cardinality_from_name(name)
        self.models = self.load_models(SCATS_DATA.get_location_id(name), ["gru"])

    def load_models(self, name, model_names):
        models = {}
        for model_name in model_names:
            try:
                with open('predictedvalues.json', 'r') as openfile:
 
                    # Reading from json file
                    json_object = json.load(openfile)
                models[model_name] = json_object[model_name][str(self.node.scats_number)][str(name)]
            except Exception as e:
            	print("{0} model for junction {1} could not be found!".format(model_name, name))
        return models

    def contains_street(self, street_name):
        return street_name in self.streets


class Graph(object):

    def __init__(self):
        self.nodes = []

    def get_path(self, origin, destination, restrictions):
        path = [(self.get_node(origin), self.get_node(origin).incoming_connections[0])]
        path, restrictions = self.find_next_best_node(path, self.get_node(destination), 0, restrictions)
        return path, restrictions

    def find_next_best_node(self, path, destination, index, restrictions):
        # print(len(path[index][0].outgoing_connections))
        restrictions.append([])
        for connection in path[index][0].outgoing_connections:
            if connection.node == destination:
                # print("found destination")
                path.append((connection.node, connection))
                restrictions[index].append(path[index][0])
                return path, restrictions
            elif distance(connection.node.coordinates, destination.coordinates) < distance(path[index][0].coordinates, destination.coordinates):

                # print("found node {0} closer to destination".format(connection.node.scats_number))
                try:
                    if connection.node in restrictions[index + 1]:
                        # print("node restricted")
                        return path, restrictions
                except:
                    pass
                path.append((connection.node, connection))
                return self.find_next_best_node(path, destination, index + 1, restrictions)
        return path, restrictions

    def add_node(self, node):
        self.nodes.append(node)

    def get_node(self, scats_number):
        return next(x for x in self.nodes if x.scats_number == scats_number)

    def show_graph(self):
        for node in self.nodes:
            print("{0} - {1} {2}\nConnections:".format(node.scats_number, node.incoming_connections[0].streets[0], node.incoming_connections[0].streets[1]))
            for connection in node.outgoing_connections:
                print("\t{0} - {1} {2}".format(connection.node.scats_number, connection.streets[0], connection.streets[1]))
            print("\n")

    def get_paths(self, origin, destination, min_path_count, model, time_in_minutes):
        paths = []
        restrictions = []
        for x in range(min_path_count):
            path, restrictions = self.get_path(origin, destination, restrictions)
            if path[-1][0].scats_number != destination:
                print("\nNo more alternative paths.")
                return
            paths.append(path)
            print("=====")
            total_cost = 0
            elapsed_time = 0
            index = 0
            for i, j in path:
                if index+1 == len(path):
                    break
                if index == 0:
                    print("Origin: {0} - {1} {2}.".format(i.scats_number, j.streets[0], j.streets[1]))
                else:
                    time_index = math.floor((time_in_minutes+elapsed_time)/15)
                    print(time_index)
                    volume = j.models[model][time_index]
                    total_cost += volume
                    distance_in_km = distance(i.coordinates, path[index+1][0].coordinates)
                    time = self.calculate_time(volume, 60, distance_in_km)
                    print("{0} - {1} {2}. Cost: {3:.2f} mins Distance {4:.2f}km".format(i.scats_number, j.streets[0], j.streets[1], time*60, distance_in_km))
                    elapsed_time += time*60
                index += 1

            print ("\n\t Total time to destination: {0:.0f} mins {1} seconds".format(elapsed_time, decimal_to_seconds(elapsed_time-math.floor(elapsed_time))))

    def calculate_time(self, volume, speed_limit, distance):
        travel_speed = get_speed_coefficient(volume)
        print("Speed: {0}. Volume {1}".format(travel_speed, volume))
        return distance/travel_speed

def get_speed_coefficient(C):
    A = -0.9765625
    B = 62.5
    speed = 60    
    D = pow(B, 2) - (4*A*-C)
    return np.clip(((-B-sqrt(D)/(2*A))+94.5), 0, 60)

def decimal_to_seconds(value):
    return math.floor((value/100)*6000)

def get_graph():
    graph = Graph()

    for scats in SCATS_DATA.get_all_scats_numbers():
        coordinates = SCATS_DATA.get_positional_data(scats)
        node = Node(scats, coordinates)
        print("adding connections for {0}".format(scats))
        for approach in SCATS_DATA.get_scats_approaches_names(scats):
        	node.add_incoming_connection(Connection(approach, node))
        graph.add_node(node)

    for node in graph.nodes:
        node.find_outgoing_connections(graph)
    return graph


def main():
    graph = get_graph()
    graph.get_paths(970, 4040, 5, "gru", 0*4*15)


if __name__ == '__main__':
    main()
