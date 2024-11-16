
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import networkx as nx
from networkx.readwrite.json_graph import cytoscape_data
from networkx.readwrite.pajek import write_pajek
from networkx.readwrite import json_graph
import json
import pandas as pd
from os import getcwd
import re

import pyodbc

from matplotlib import pyplot as plt

from Utils.ORD_Static import Static 
from Utils.ORD_ConsultUtils import ConsultUtils as Utils


class ConsultNetwork(object):
    """
    A set of Methods to build a network of 'who consults who' in a referral Network
    """

    def __init__(self, db, sta3n):
        '''
        A method to set a network
        '''
       
        self.sta3n = sta3n 
        s = Static(db)
        sql = s.get_sql(sta3n)
        self.data = s.get_data(sta3n = sta3n)
        

        # self.source = self.data.FromPrimaryStopCode.dropna().unique()
        # self.target = self.data.ToPrimaryStopCode.dropna().unique()

        self.source = self.data.ConsultFromPrimaryStopCode.dropna().unique()
        self.target = self.data.VisitAtPrimaryStopCode.dropna().unique()

        self.G = nx.DiGraph()
        self.get_t2appt_node()
        self.get_code2name()
        self.make_graph()


    def format_stopcode(self, stopcode):
        '''
        Transform the stop code into a int and from that to a string, if NA return NA
        stopcode - accepts a stop code as parameter.
        '''
        if (pd.isnull(stopcode)):
            return None
        else:
            return str(int(stopcode))

        


    def get_code2name(self):
        '''
        method to return a dictionary {StopCode:Name}
        '''
        
        f = { row[1]['ConsultFromPrimaryStopCode']:row[1]['ConsultFromPrimaryStopCodeName'] for row in \
                self.data[['ConsultFromPrimaryStopCode', 'ConsultFromPrimaryStopCodeName',]].drop_duplicates().dropna().iterrows()} # source
        target = self.data[['VisitAtPrimaryStopCode','VisitAtPrimaryStopCodeName']].drop_duplicates().dropna()
        
        t = {r[1]['VisitAtPrimaryStopCode']:r[1]['VisitAtPrimaryStopCodeName'] for r in target.iterrows()}
        f.update(t)
        self.code2name = f
        
        
   


    def edge_count(self,source,target):
        """
        A method to return the number of requests made from 'source' to a 'target'
        """ 
        return self.data[(self.data.FromPrimaryStopCode==source) & (self.data.ToPrimaryStopCode==target)].shape[0]


    
    def get_edge_weights(self,source,target):
        """
        A method to get the weight for the edges the weights being the median time for processing of requests.
        """
        df = self.data[(self.data.ConsultFromPrimaryStopCode==source) & (self.data.VisitAtPrimaryStopCode==target)]
        
        return df.DeltaTime

    def get_nodes(self):
        """
        A method to return all nodes
        """
        fromNodes = self.data.ConsultFromPrimaryStopCode.unique()
        toNodes =  self.data.VisitAtPrimaryStopCode.unique()
        return [{'name':int(n)} for n in np.concatenate((fromNodes,toNodes)) if not pd.isnull(n)]

    def get_edges(self):
        """
        A method to return all edges for the graph
        """
        el = []
        edges = {f:self.data[self.data.ConsultFromPrimaryStopCode==f][['ConsultFromPrimaryStopCode','VisitAtPrimaryStopCode']].groupby('VisitAtPrimaryStopCode').count() for f in self.source}

        for k,v in edges.items():
            for target in v.index:
                el.append({"source":int(k),
                        "target":int(target),
                        "source_name":self.code2name[k],
                        "target_name":self.code2name[target],
                        "weight_time": self.get_edge_weights(k,target).median(),
                        "traffic_time":self.get_edge_weights(k,target).to_list(),
                        'count':v.loc[target].ConsultFromPrimaryStopCode,
                        'scale_count' : 0.01*v.loc[target].ConsultFromPrimaryStopCode
                })
        return  el


    def get_t2appt_node(self):
        '''
        Get times to appt in a given clinic
#         '''

        # start_time = self.data.AppointmentDateTime.apply(lambda x : pd.to_datetime(x))
        # end_time = self.data.RequestDateTime.apply(lambda x : pd.to_datetime(x))
        # self.data['t2appointment'] = (start_time-end_time).apply(lambda x: x.days)
        self.t2Appt = {}
        for t in self.target:
           self.t2Appt[t]  = self.data[self.data.VisitAtPrimaryStopCode == t]['DeltaTime'] # get the whole vector

       

    def get_t2app_location(self):
        '''
        Give the time for appointment by clinic location
        '''
        a = self.data.AppointmentDateTime.apply(lambda x : pd.to_datetime(x))
        b = self.data.ConsultRequestDate.apply(lambda x : pd.to_datetime(x))
        self.data['t2app_loc'] = (a-b).apply(lambda x: x.days)
        t2Appt_loc = {}
        for l in self.data.ConsultToLocation.unique():
            t2Appt_loc[l] = self.data[self.data.ConsultToLocation ==l][t2Appt_loc].drop_duplicates()
        return t2Appt



    def mission (self,t2appt):
        '''
        A method to return the  MEDIAN and IQR of wait time
        '''
        plt.rcParams.update({'font.size': 7})
        t2app_dict = {k: t2appt[k][t2appt[k]>=0] for k in t2appt}
        mission = pd.DataFrame().from_dict({self.code2name[str(k)]:t2appt[k].quantile([0.25,0.5,0.75]).to_list() for k in t2app_dict})
        mission = mission.T
        mission.columns = ['lower','median','upper']
        mission = mission.sort_values('median')
        median = mission['median']
        err = [mission['lower'].values, mission['upper'].values]
        fig, ax  = plt.subplots(1,1)
        median.plot(kind = 'barh',xerr = err, ax= ax,figsize = (12,8))
        ax.grid('on')
        ax.set_xlabel('days')
        ax.set_title('time to appointment by stopcode YTD')
        plt.show()
        plt.savefig('t2apptStopCode.png')
        mission.sort_values('median').to_csv('t2appt_StopCode.csv')
        return mission.sort_values('median')

    def json_dump(self):
        """
        A method to dump the json of the nodes and links
        """     
        graphFile = {'nodes':self.get_nodes(),'links':self.get_edges()}
        with open('graphFile.json','w') as fp:
            json.dump(graphFile,fp)

    def make_graph(self):
        """
        A method to return a graph object
        """
        nodes = [n['name'] for n in self.get_nodes()]    
        el = [(e['source'],e['target'],e['source_name'],e['target_name'],e['weight_time'],e["traffic_time"],e['count']) for e in self.get_edges()]

        self.G.add_nodes_from(nodes)
       
        for s,t,sn,tn,w,traffic_time,c in el:
            self.G.add_edge(s,t,sourceName = sn,targetName = tn, median_day=w ,traffic_time = traffic_time,  count=int(c))
       
        for ix,indeg in self.G.in_degree():
            self.G.nodes[ix]['width'] = indeg
            self.G.nodes[ix]['height'] = indeg
            self.G.nodes[ix]['name'] = self.code2name[ix]

        for ix in self.G.nodes():
            try:
                self.G.nodes[ix]['color'] = int(self.t2Appt[ix].median()) # color by time to appointment
                self.G.nodes[ix]['t2appt'] = list(self.t2Appt[ix])
            except:
                self.G.nodes[ix]['color'] = int(0.0)
                self.G.nodes[ix]['t2appt'] = [0,0,0,0,0]

   
      

# run
con_nw = ConsultNetwork('ORD',695)
utils = Utils()
utils.load(con_nw)
utils.json_graph_save()
