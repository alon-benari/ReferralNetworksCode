import numpy as np
import Utils.ORD_Analysis as a
import matplotlib.pyplot as plt
import Utils.ORD_ConsultUtils as u
import Utils.ORD_Static  as s
import Utils.ORD_ConsultNetwork as c
import networkx as nx

from collections import Counter

class Triangles:
    '''
    A method to look into triangle in the network, and their statistics
    '''
    utils = u.ConsultUtils()
    static = s.Static('ORD')
    gr = utils.load_graph_iter() # load a bunch of graphs as a dictionary
    G = nx.Graph() # undirected graph

    def no_selfedge(self,G):
        '''
        A method to remove self edge and return a trimmed Graph
        '''
        
        for n in G.nodes():
            if G.has_edge(n,n):
                G.remove_edge(n,n)
        return G

    def get_triangle(self, G,primary_node, direct = 1):
        '''
        A method to return a list of triangles in a network, for a given node
        '''
        G = self.no_selfedge(G)
        if direct == 0:
            G = G.to_undirected()
        #
        tri = list()
        node1 = [i for i in G.successors(primary_node)]  # find the primary Node  successors
       
        children = lambda node : [i for i in G.successors(node)]
        for n1 in node1:
            node2 = children(n1)    # get children of node 2
            for n2 in node2:
                if G.has_edge(n2,primary_node):
                    tri.append([primary_node,n1,n2])
        return  tri #list(map(np.sort, tri))

    def triangle_graphs(self):
        '''
        A method to return a list of non -directed triangles for a given graph
        '''
        triangle_dict = dict()
        for sta3n  in list(self.gr.keys()):
            G = self.gr[sta3n]
            nodes = G.nodes()
            triangle_list = list()
            for n in nodes:
                triangle_list.extend(self.get_triangle(G,int(n)))
            triangle_dict[sta3n] = [tuple(np.sort(e)) for e in triangle_list]
        return triangle_dict

    def count_by(self, iterable):
        '''
        A method to return a dictionary of  items and how many are there of each kind, kind of group by
        '''

    def mash_triangles(self):
        '''
        A method to return a set of all existing triangles , the iterable will come form self.triangle_graphs
        returns a ranked triangle set, i.e. which  triangles are shown the most.
        '''
        # bag = self.triangle_graphs()
        tri_bag = []
        for  t in self.triangle_graphs().values():
            tri_bag.extend(t)
        count = Counter(tri_bag)
        tri_rank = sorted(count.items(), key = lambda x:x[1], reverse = True)
        return tri_rank
    

    def get_edges(self, tuple):
        '''
        A method to return a list of tuples to serve as the list of edges between the nodes
        '''
        return [(tuple[0],tuple[1]),(tuple[1],tuple[2]),(tuple[0], tuple[2])]

    def render_tri_edges(self):
        '''
        A method to return a network composed of edges  between clinics that are part of triangles
        '''
        rank = self.mash_triangles()
        edges = list()
        for x in rank:
            edges.extend(self.get_edges(x[0]))

        return self.G.add_edges_from(edges) 
       

    

    


tri = Triangles()
# sta3n = '640'
# name = lambda x : gr[sta3n].nodes()[x]['name']

# [[list(map(lambda n: name(n), trng)), count] for trng, count in rank[:50]]


