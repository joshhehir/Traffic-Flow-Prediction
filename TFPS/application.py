from data.scats import ScatsData
import math
import numpy as np

SCATS_DATA = ScatsData()



def angle_of_vectors(a,b,c,d):
    
     dotProduct = a*c + b*d
         # for three dimensional simply add dotProduct = a*c + b*d  + e*f 
     modOfVector1 = math.sqrt( a*a + b*b)*math.sqrt(c*c + d*d) 
         # for three dimensional simply add modOfVector = math.sqrt( a*a + b*b + e*e)*math.sqrt(c*c + d*d +f*f) 
     angle = dotProduct/modOfVector1
     angleInDegree = math.degrees(math.acos(angle))
     #print("θ =",angleInDegree,"°")
     return angleInDegree

def get_streets_from_name(name):
	words = name.upper().split(' OF ')
	streetB = words[1]
	streetA = words[0].split()
	streetA = ' '.join(streetA[:len(streetA)-1])
	return [streetA, streetB]

def get_cardinality_from_name(name):
	words = name.upper().split(' OF ')
	cardinality = words[0].split()
	cardinality = cardinality[len(cardinality)-1:][0]
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
		return (0,1)
	elif cardinality == 1:
		return (1,1)
	elif cardinality == 2:
		return (1,0)
	elif cardinality == 3:
		return (1,-1)
	elif cardinality == 4:
		return (0,-1)
	elif cardinality == 5:
		return (-1,-1)
	elif cardinality == 6:
		return (-1,0)
	elif cardinality == 7:
		return (-1,1)

def get_inverse_cardinality(cardinality):
	reverse = cardinality + 4
	if reverse >= 8:
		reverse = reverse - 8
	return reverse

def distance(origin, target):
	return abs(math.dist(origin, target))

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
		"""
		1. get connections of connected nodes
		2. get nodes in direction of cardinality
		3. get closest node
		"""
		valid_connections = []
		for node in graph.nodes:
			for external_connection in node.incoming_connections:
				if external_connection.streets[0] in connection.streets or external_connection.streets[1] in connection.streets:
					if external_connection.streets[0] in connection.streets and external_connection.streets[1] in connection.streets:
						#print("{0} and {1} present in connection".format(external_connection.streets[0], external_connection.streets[1]))
						None
					else:
						vector_a = convert_cardinality_to_vector(connection.cardinality)
						vector_b = tuple(map(lambda i, j: i-j, external_connection.node.coordinates, connection.node.coordinates))
						#print ("{0}\nCardinality: {1}. Direction: {2}".format(external_connection.streets, vector_a, vector_b))
						if angle_of_vectors(vector_a[1], vector_a[0], vector_b[0], vector_b[1]) < 45:
							valid_connections.append(external_connection)
				
		print("Node: {0}\nIncoming Connection: {1} {2} {3}\nValid Connections:".format(self.scats_number, connection.streets[0], connection.cardinality, connection.streets[1]))
		best_connection = None
		for external_connection in valid_connections:
			print("\t{0} {1} {2}. Distance = {3}".format(external_connection.streets[0], external_connection.cardinality, external_connection.streets[1], distance(connection.node.coordinates, external_connection.node.coordinates)))
			if best_connection == None:
				best_connection = external_connection
			elif (distance(connection.node.coordinates, external_connection.node.coordinates) < distance(connection.node.coordinates, best_connection.node.coordinates)):
				best_connection = external_connection
			elif (distance(connection.node.coordinates, external_connection.node.coordinates) == distance(connection.node.coordinates, best_connection.node.coordinates)):
				if abs(get_inverse_cardinality(connection.cardinality) - external_connection.cardinality) < abs(get_inverse_cardinality(connection.cardinality) - best_connection.cardinality):
					best_connection = external_connection
		if best_connection != None:
			print("Best Connection = {0} {1} {2}. Distance = {3}\n".format(best_connection.streets[0], best_connection.cardinality, best_connection.streets[1], distance(connection.node.coordinates, best_connection.node.coordinates)))
		else:
			print("\tThere are no connections for this node")
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
		self.models = {}


	def load_models(self, name, model_names):
		models = {}
		for model_name in model_names:
			try:
				models[model_name] = ("model/{0}/{1}/{2}.h5".format(model_name, node.scats_number, name))
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
		#print(len(path[index][0].outgoing_connections))
		restrictions.append([])
		for connection in path[index][0].outgoing_connections:
			if connection.node == destination:
				#print("found destination")
				path.append((connection.node, connection))
				restrictions[index].append(path[index][0])
				return path, restrictions
			elif distance(connection.node.coordinates, destination.coordinates) < distance(path[index][0].coordinates, destination.coordinates):
				
				#print("found node {0} closer to destination".format(connection.node.scats_number))
				try:
					if connection.node in restrictions[index+1]:
						#print("node restricted")
						return path, restrictions
				except:
					pass
				path.append((connection.node, connection))
				return self.find_next_best_node(path, destination, index+1, restrictions)
		return path, restrictions

	def add_node(self, node):
		self.nodes.append(node)

	def get_node(self, scats_number):
		return next(x for x in self.nodes if x.scats_number == scats_number)

def show_graph():
	for node in graph.nodes:
		print("{0} - {1} {2}\nConnections:".format(node.scats_number, node.incoming_connections[0].streets[0], node.incoming_connections[0].streets[1]))
		for connection in node.outgoing_connections:
			print("\t{0} - {1} {2}".format(connection.node.scats_number, connection.streets[0], connection.streets[1]))
		print("\n")

def main():
	graph = Graph()

	for scats in SCATS_DATA.get_all_scats_numbers():
		coordinates = SCATS_DATA.get_positional_data(scats)
		node = Node(scats, coordinates)
		for approach in SCATS_DATA.get_scats_approaches(scats):
			node.add_incoming_connection(Connection(approach, node))
		graph.add_node(node)

	for node in graph.nodes:
		node.find_outgoing_connections(graph)
	print("\n\n\n")

	#show_graph()

	restrictions = []
	paths = []
	min_path_count = 5
	origin = 4324
	destination = 4262 
	for x in range(min_path_count):
		path, restrictions = graph.get_path(origin, destination, restrictions)
		if (path[-1][0].scats_number != destination):
			print("\nNo more alternative paths.")
			return
		paths.append(path)
		print("=====")
		for i, j in path:
			print("{0} - {1} {2}".format(i.scats_number, j.streets[0], j.streets[1]))



if __name__ == '__main__':
	main()