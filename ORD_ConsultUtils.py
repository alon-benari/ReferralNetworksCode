import networkx as nx
import json
import pyodbc
import pandas as pd 
import numpy as np
from networkx.readwrite.json_graph import cytoscape_data, cytoscape_graph
from networkx.readwrite.json_graph import node_link_data, node_link_graph
import os
import re
from networkx.readwrite import json_graph


class ConsultUtils(object):
    '''
    A set of methods to process the consult network graph

    cyclic_path : return  a path(s) of size N
    get_cytoscape: return a cytoscape object from the graph object
    json_it: save a json out of the  cytoscape object

    '''
    def __init__(self):
        '''
        Initialize the class with a graph instance coming from ORD_ConsultNetwork.
        Params:

        graph_obj - an ORD_ConsultNetwork.
        '''

        
        
        
    def load(self,graph_obj):
        '''
        A method to load an object graph
        
        Object graph -  a graph object like the one
        '''
        self.graph_obj = graph_obj
        self.G = graph_obj.G
        self.code2name = graph_obj.code2name
        self.sta3n = graph_obj.sta3n
         


   
    def get_cytoscape(self):
        '''
        A method to output a cytoscape object
        '''
        return cytoscape_data(self.G)['elements']
    

    def get_cytoscape(self, node, func):
        '''
        A method to output a cytoscape object of a subgraph of 
        '''
        sub_G = nx.subgraph(self.G, func( node))
        return cytoscape_data(sub_G)['elements']
    

    def json_graph_save(self):
        """
        Dump a json of data into a file
        """
        # data = node_link_data(self.G)
        data = self.get_cytoscape()
        #graph_name = str(self.graph_obj.sta3n)+'graph_version0.json'
        graph_name = str('612')+'graph_version0.json'
        print(graph_name)
        
        try:
            os.chdir('./Graphs')
            with open(graph_name,'w') as f:
                json.dump(data,f, indent = 4)
                os.chdir('..')
        except:
            print("An error occured, could not save the graph")
           
    
    def json_graph_save(self, node, func, title):
        """
        Dump a json of data into a file, using the overloaded get_cytoscape method
        """
        # data = node_link_data(self.G)
        data = self.get_cytoscape(node, func)
        #graph_name = str(self.graph_obj.sta3n)+'graph_version0.json'
        graph_name = title+'graph_version0.json'
        print(graph_name)
        
        try:
            os.chdir('./Graphs')
            with open(graph_name,'w') as f:
                json.dump(data,f, indent = 4)
            os.chdir('..')
        except:
            print("An error occured, could not save the graph")





    



    def flatten(self, list_of_lists):
        '''
        A method to return a list from a list of lists
        '''
        empty_list = []
        for l in list_of_lists:
            empty_list.extend(l)
        return empty_list

    def save_iter_graphs(self):
        '''
        A method to iterate over all CSVs, and create and save a graph from them in /Graphs
        '''
        os.chdir('./Assets')
        graphs = [ ConsultNetwork(f) for f in os.listdir()]
        os.chdir('..')
        Utils_obj = [self.load(g) for g in graphs]
        for obj in Utils_obj:
            Utils.json_graph_save()
        return graphs

    def parse_sta3n(self, fname):
        '''
        A method to return the sta3n from a file name
        fname -  string, filename to be parsed
        '''
        sta3n = re.findall("[0-9][0-9][0-9]", fname)[0]
        return sta3n

    def load_graph(self,json_fn):
        '''
        A method to return a json file name into networkx object
        json_fn =  string, .json file name
        
        '''
        try:
            with open(json_fn) as f:
                
                data =  json.load(f)
                return json_graph.cytoscape_graph(data)
        except:
            print('An error occured, could not parse load file')


        

    def load_graph_iter(self):
        '''
        A method to return an iterable of graph objects 

        
        '''
        os.chdir('./Graphs')
        try:
            graphs = {self.parse_sta3n(js):self.load_graph(js)  for js in os.listdir()}
        except e:
            print(e)
        os.chdir('..')
        return graphs



#utils = ConsultUtils()
#gr = utils.load_graph_iter() # load a bunch of graphs as a dictionary

    
