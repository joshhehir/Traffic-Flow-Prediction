from data.scats import ScatsData
from math import radians, cos, sin, asin, sqrt, degrees, acos, floor
import json
import numpy as np
from keras.models import load_model

SCATS_DATA = ScatsData()

class Vector(object):

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def angle(self, other):
        '''Angle of vectors
            Calculate the angle between self and other vector
            # Arguements
                other: 2D vector, vector to check angle against
            # Returns
                angle_in_degrees: Float, angle between provided vectors
        ''' 

        dot_product = self.x * other.x + self.y * other.y
        mod_of_vector = sqrt(self.x**2 + self.y**2) * sqrt(other.x**2 + other.y**2)
        angle = dot_product / mod_of_vector
        angle_in_degrees = degrees(acos(angle))
        return angle_in_degrees

    def distance(self, other):
        '''Distance
            Calculate distance to other vector 
            # Arguements
                other: 2D vector
            # Returns
                result: Float, distance from self to other
        '''

        lon1 = radians(self.x)
        lon2 = radians(other.x)
        lat1 = radians(self.y)
        lat2 = radians(other.y)
          
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
     
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in kilometers. Use 3956 for miles
        r = 6371
          
        # calculate the result
        return(c * r)


    def direction(self, other):
        '''Direction
            Calculate direction to other vector
            # Arguements
                other: 2D vector
            # Returns
                result: Float, direction to other vector
        '''


        return Vector(self.x - other.x, self.y - other.y)

    def __sub__(self, other):
        return Vector(other.x-self.x, other.y-self.y)

    def __neg__(self):
        return Vector(-self.x, -self.y)

    def __str__(self):
        return "({0}, {1})".format(self.x, self.y)

class Node(object):

    def __init__(self, scats_number, coordinates):
        self.scats_number = scats_number
        self.coordinates = Vector(coordinates[0], coordinates[1])
        self.incoming_connections = []
        self.outgoing_connections = []

    def find_outgoing_connections(self, graph):
        '''Find outgoing connections
            Searches through incoming connections to find outgoing connections
            # Arguements
                graph: Node List, represents the node graph
            # Returns
                None
        '''

        for connection in self.incoming_connections:
            connection = self.get_respective_outgoing_connection(connection, graph)
            if connection != None:
                self.outgoing_connections.append(connection)

    def get_respective_outgoing_connection(self, connection, graph):
        '''Get respective outgoing connection
            Gets an incomming connection from another node which is the inverse of the provided connection
            # Arguements
                connection: Connection, the incoming connection
                graph: Node List, represents the node graph
            # Returns
                best_connection: the connection most probable to be the inverse of the provided connection
        '''

        # Find all valid connections
        valid_connections = []
        for node in graph.nodes:
            for external_connection in node.incoming_connections:
                if self.connections_share_one_street(connection, external_connection):
                    if self.connection_within_angle_range(connection, external_connection):
                        valid_connections.append(external_connection)

        if len(valid_connections) < 1:
            return
        # Select best connection
        best_connection = valid_connections[0]
        origin = connection.node.coordinates
        for external_connection in valid_connections:
            
            current_position = external_connection.node.coordinates
            best_position = best_connection.node.coordinates

            if origin.distance(current_position) < origin.distance(best_position):
                best_connection = external_connection
            elif (origin.distance(current_position) == origin.distance(best_position)):
                if origin.angle(-current_position) < origin.angle(-best_position):
                    best_connection = external_connection
        return best_connection

    def connections_share_one_street(self, connection_a, connection_b):
        if connection_a.contains_streets_count(connection_b.streets) == 1:
            return True
        return False

    def connection_within_angle_range(self, connection_a, connection_b):
        vector_a = connection_a.direction
        vector_b = connection_b.node.coordinates-connection_a.node.coordinates
        if vector_a.angle(vector_b) < 45:
            return True
        return False

class Connection(object):

    def __init__(self, name, node):
        self.node = node
        self.streets = self.get_streets_from_name(name)
        self.direction = self.get_vector_from_name(name)
        self.models = []

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

    def contains_streets_count(self, streets):
        intersect_count = 0
        for street in streets:
            if street in self.streets:
                intersect_count += 1
        return intersect_count

    def get_vector_from_name(self, name):
        words = name.upper().split(' OF ')
        direction = words[0].split()
        direction = direction[len(direction) - 1:][0]
        if direction == "N":
            return Vector(0,1)
        elif direction == "NE":
            return Vector(1,1)
        elif direction == "E":
            return Vector(1,0)
        elif direction == "SE":
            return Vector(1,-1)
        elif direction == "S":
            return Vector(0,-1)
        elif direction == "SW":
            return Vector(-1,-1)
        elif direction == "W":
            return Vector(-1, 0)
        elif direction == "NW":
            return Vector(-1, 1)
        else:
            return None

    def get_streets_from_name(self, name):
        words = name.upper().split(' OF ')
        streetB = words[1]
        streetA = words[0].split()
        streetA = ' '.join(streetA[:len(streetA) - 1])
        return [streetA, streetB]


class Graph(object):

    def __init__(self):
        self.nodes = []

    def get_path(self, origin, destination, restrictions):
        path = [(self.get_node(origin), self.get_node(origin).incoming_connections[0])]
        path, restrictions = self.find_next_best_node(path, self.get_node(destination), 0, restrictions)
        return path, restrictions

    def find_next_best_node(self, path, destination, index, restrictions):
        print(len(path[index][0].outgoing_connections))
        restrictions.append([])
        for connection in path[index][0].outgoing_connections:
            if connection.node == destination:
                print("found destination")
                path.append((connection.node, connection))
                restrictions[index].append(path[index][0])
                return path, restrictions
            elif connection.node.coordinates.distance(destination.coordinates) < path[index][0].coordinates.distance(destination.coordinates):

                print("found node {0} closer to destination".format(connection.node.scats_number))
                try:
                    if connection.node in restrictions[index + 1]:
                        print("node restricted")
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
                    time_index = floor((time_in_minutes+elapsed_time)/15)
                    volume = j.models[model][time_index]
                    total_cost += volume
                    distance_in_km = distance(i.coordinates, path[index+1][0].coordinates)
                    time = self.calculate_time(volume, 60, distance_in_km)
                    print("{0} - {1} {2}. Cost: {3:.2f} mins Distance {4:.2f}km".format(i.scats_number, j.streets[0], j.streets[1], time*60, distance_in_km))
                    elapsed_time += time*60
                index += 1

            print ("\n\t Total time to destination: {0:.0f} mins {1} seconds".format(elapsed_time, decimal_to_seconds(elapsed_time-floor(elapsed_time))))

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
    return floor((value/100)*6000)

def get_graph():
    graph = Graph()

    for scats in SCATS_DATA.get_all_scats_numbers():
        coordinates = SCATS_DATA.get_positional_data(scats)
        node = Node(scats, coordinates)
        print("adding connections for {0}".format(scats))
        for approach in SCATS_DATA.get_scats_approaches_names(scats):
            node.incoming_connections.append(Connection(approach, node))
        graph.add_node(node)

    for node in graph.nodes:
        node.find_outgoing_connections(graph)
    return graph


def main():
    graph = get_graph()
    graph.show_graph()
    graph.get_paths(970, 4040, 5, "gru", 0*4*15)


if __name__ == '__main__':
    main()
