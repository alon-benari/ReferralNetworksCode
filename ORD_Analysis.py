import Utils.ORD_ConsultUtils as u
import Utils.ORD_Static  as s
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Utils.PyNorCal as pync
import networkx as nx
import seaborn as sns
import powerlaw
import plotly.express as px
from Utils.PyNorCal import PowerLaw
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler





class Analysis:
    '''
    A set of methods to assist in analysis
    '''

    #def __init__(self):
    '''
    (1) create a dictionary of sta3ns and graph objects
    (2) Load the Static class the static.sta3n_counties.
    '''
    utils = u.ConsultUtils()
    static = s.Static('ORD')
    static.get_vet_counts()
    graph_sta3n = utils.load_graph_iter() # load a bunch of graphs as a dictionary
   

    sta3n_population = static.vet_count


    def show_sna_stats(self,key_string, func ):
        '''
        A method to return a plot of a metric as a function of Sta3n population.
        func -  function to pass for data processing( mean, median etc)
        '''

        data =[[k,  self.sta3n_population.get(k)['vet_count'], func(self.sna_stats(k)[k][key_string])]  for k in self.graph_sta3n.keys()]
        #pd.DataFrame.from_dict(data, orient='index',columns = ['population',key_string]).plot.scatter(x='population',y=key_string,loglog = True)
        #plt.show()
        return pd.DataFrame(data, columns = ['sta3n','population',key_string]).set_index('sta3n')

    def sna_stats(self, sta3n):
        """
        A method to return an object with multiple basic SNA statistics
        """
        gr = self.graph_sta3n[sta3n]
        
        return {
                sta3n:{
                'total_appointments':np.array([gr.edges()[(s,t)]['count']  for s,t in gr.edges()]),
                'total_times': self.utils.flatten([gr.edges()[(s,t)]['traffic_time'] for s,t in gr.edges()]),
                'cluster':nx.average_clustering(gr),
                'transitivity': nx.transitivity(gr),
                'node_number':gr.number_of_nodes(),
                'in_degree':[d[1] for d in gr.in_degree()],
                'dict_in_degree': dict(gr.in_degree()),
                'out_degree':[d[1] for d in gr.out_degree()],
                'dict_out_degree': dict(gr.out_degree()),
                'page_rank': nx.pagerank(gr),
                'closeness':nx.closeness_centrality(gr),
                'btw_centrality':sorted(nx.betweenness_centrality(gr).items(), key = lambda kv:(kv[1])),
                'density':nx.density(gr),
                'edge_btw_cent':sorted(nx.edge_betweenness_centrality(gr, weight = 'count').items() ,key = lambda kv: (kv[1])),
                

                
                
                    }
                }
    def get_sigma(self, sta3n):
        '''
        A  method to return the sima coefficient to determine Small World  network
        '''
        gr = self.graph_sta3n[sta3n]
        return {
            sta3n: nx.sigma(gr.to_undirected(), niter = 5, nrand = 5 )
        }

    def save_sigma(self):
        
        data = []
         
        for k,v, in list(self.graph_sta3n.items())[31:60]:
            try:
                print({k:self.get_sigma(k)})
                data.append(self.get_sigma(k))
            except Exception as e:
                
                print({k:e})

        try:
            os.chdir('./Static')
            with open('sigma.json','w') as f:
                json.dump(data,f, indent = 4)
                os.chdir('..')
                return data

        except:
            print("An error occured, could not save the graph")
            return data






    def edge_stats(self, sta3n):
        '''
        a method to return edge related stats
        '''
        gr =self.graph_sta3n.get(sta3n)
        
        edge_btw = pd.DataFrame.from_dict(
                            nx.edge_betweenness_centrality(gr), orient = 'index', columns =['edge_btw']
                            )
        edge_time = pd.DataFrame.from_dict(
                    { (s,t): np.median(gr.edges()[s,t].get('traffic_time')) for (s,t) in gr.edges()},
                     orient = 'index', columns =['edge_time']
                     )
        edge_vol = pd.DataFrame.from_dict(
                    { (s,t): gr.edges()[s,t].get('count') for (s,t) in gr.edges()},
                     orient = 'index', columns =['edge_count']
                     )        
        
        return edge_btw, edge_vol, edge_time

    def quick_plot(self,xx, yy):
        '''
        A method to quickly render a plot
        '''
        fig, ax = plt.subplots(figsize=(14,10))
        ax.scatter(np.log10(xx),np.log10(yy))
        #ax.hist(data, bins = range(1,50))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('population')
        ax.set_ylabel('#consults')
        #pw = PowerLaw()
        #res = pw.fit_model(np.log(yy),np.log(xx))
        #X= np.linspace(1,200000,200)
        #ax.plot(np.log10(X),res.params[1]*np.array(np.log10(X))+res.params[0])
        fig.savefig('./notes/ScattrPlotPW.eps')
        fig.show()

   

        

    def strong_cc(self):
        '''
        A method to return the largest connected components in the graphs for each network
        '''
        # strongly connectec component
        cc =  {k: [v.subgraph(c)  for c in nx.strongly_connected_components(v) if  len(c)>10][0]  for k,v  in gr.items()}  # keyed dictionary of CC

        cc_node_count = {k:len(v[0]) for k,v  in cc.items()}

        return cc

   


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

    



    def cyclic_path(self, G, path):
       '''
       A method to return  cyclic paths
       path - integer , path is the length of the path 
       G -  DiGraph Object
       '''
       cyc0 = []
        
       paths =tuple([c for  c in nx.simple_cycles(G) if len(c) == path])
    #    for p in paths:
    #       cyc0.append(list(map(lambda x: self.code2name[str(x)],p)))
            
       return paths


    


    def scale_total_appointments(self):
        '''
        A method to return the x and y axis of a plot of a model
        '''
        keys =list(self.graph_sta3n.keys())[1:]
        
        xx = [self.static.vet_count[k]['vet_count'] for k in keys] # population size of sta3n
        yy = [self.sna_stats(k)[k]['total_appointments'].sum() for k in keys]

        return xx, yy

    
    def scale_transitivity(self):
        '''
        A method to return the x and y axis of a plot of a model
        '''
        keys =list(self.graph_sta3n.keys())[1:]
        # works good.
        #xx = [self.sna_stats(k)[k]['transitivity'] for k in keys]# population size of sta3n
        #yy = [self.sna_stats(k)[k]['total_appointments'].sum() for k in keys]

        ##works good
    

    def graph_features(self, load = 1):
        '''
        return a dataset with the features
        '''
        normal = lambda x: (np.array(x)-np.array(x).mean())/np.array(x).std() # standardize

        standardz = lambda  X : np.array([ ((x-X.min())/(X.max()-X.min())) for x in X]) #normalize
        scaler = MinMaxScaler()
        wt_deg = lambda  x : np.array([es  for (n, es ) in self.graph_sta3n[x].in_degree(weight = 'count')]).mean() # get 
        if load :
            df = pd.read_csv('./Static/graph_features.csv')
            return df
        else:
            keys =list(self.graph_sta3n.keys())[1:]
            pop = [self.static.vet_count[k]['vet_count'] for k in keys]
            mean_appt = [self.sna_stats(k)[k]['total_appointments'].mean() for k in keys]
            total_appt = [self.sna_stats(k)[k]['total_appointments'].sum() for k in keys]
            transitivity = [self.sna_stats(k)[k]['transitivity']  for k in keys]
            density =   [self.sna_stats(k)[k]['density']  for k in keys]
            median_time =  [np.median(self.sna_stats(k)[k]['total_times'])  for k in keys]
            mean_time =  [np.mean(self.sna_stats(k)[k]['total_times'])  for k in keys]
            
            
            appt_norm = normal(total_appt)
            


            return pd.DataFrame({
                                    'sta3n':keys,
                                    'pop':pop,
                                    'mean_appt':mean_appt,
                                    'total_appt':total_appt,
                                    'trans':transitivity,
                                    'trans_norm':normal(transitivity),
                                    'density':density,
                                    'density_norm': normal(density),
                                    'median_time': median_time,
                                    'mean_time':mean_time,
                                    'wt_deg': [wt_deg(k) for k in  keys],
                                    'wt_deg_norm': normal([wt_deg(k) for k in  keys]),
                                    'appt_normal':appt_norm,
                                    'time_normal':normal(median_time),
                                    'pop_normal':normal(pop),
                                    'density_norm': normal(density),
                                    'stnd_wt_deg': standardz(np.array([wt_deg(k) for k in  keys])),
                                    'stnd_median_time':standardz(np.array(median_time))

                                    })




       #return xx, yy


    def scale_mean_appointments(self):
        '''
        A method to return the x and y axis of a plot of a model
        '''
        keys =list(self.graph_sta3n.keys())[1:]
        
        xx = [self.static.vet_count[k]['vet_count'] for k in keys] # population size of sta3n
        yy = [self.sna_stats(k)[k]['total_appointments'].mean() for k in keys]

        return xx, yy


    def scale_wt_deg(self):
        gr = self.graph_sta3n
        keys =list(self.graph_sta3n.keys())[1:]
        wt_deg = lambda  x : np.array([es  for (n, es ) in gr[x].in_degree(weight = 'count')]).mean()

        yy = [wt_deg(k) for k in keys]
        xx  = xx = [self.static.vet_count[k]['vet_count'] for k in keys] # population size of sta3n

        return xx, yy


    
    def scale_mean_time(self):
        '''
        A method to plot  mean time for a given HCS 

        returns the HCS patient size and the mean time for each sta3n
        '''
        keys =list(self.graph_sta3n.keys())[1:]
        
        xx = [self.static.vet_count[k]['vet_count'] for k in keys] # population size of sta3n
        yy = [np.median(self.sna_stats(k)[k]['total_times']) for k in keys]


        return xx , yy

    
    def times_2_counts(self):
        '''
        plot counts and times on a log log scale
        '''
        
        counts = self.show_sna_stats('total_appointments',np.median)
        times = self.show_sna_stats('total_times', np.median)
        data = pd.merge(counts,times)
        data.plot.scatter(x = 'total_appointments', y = 'total_times',loglog = True)
        plt.show()
        
        pw = pync.PowerLaw()
        
        
        return data, pw.fit_model(df = np.log(data.total_appointments), y = np.log(data.total_times)).summary()

    def plot_power(self, xx):
        '''
        plot density on a log log scale
        '''
        
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_yscale('log')
        h,b = np.histogram(xx, bins =100)
        ax.scatter(b[:-1],h, marker = ".", s=5)

        b, rvs = self.fit_curve(xx, stats.pareto)
        h_rvs, b_rvs = np.histogram(rvs, bins = 100)
        ax.plot(b_rvs[:-1],h_rvs, color = 'red')
        ax.set_ylabel(r'P(x)')
        ax.set_xlabel(r'x')

        # b_ln, rvs_ln = self.fit_curve(xx, scipy.stats.expon)
        # h_rvs_ln, b_rvs_ln = np.histogram(rvs_ln, bins = 100)
        # ax.plot(b_rvs_ln[:-1],h_rvs_ln, color = 'blue')



        plt.show()

    def get_coefs(self, xx, func):
        '''
        returj the scale shape and loc of the sequence
        '''
        shape, loc ,scale = func.fit(xx)
        rvs = func.rvs(shape, scale = scale, loc = loc, size = 1000)
        return [shape,loc, scale]
                   

    def fit_curve(self,xx, func):
        '''
        A method to fit the parameters for a powerlaw distribution of  density
        '''
        b,a,c = func.fit(xx)
        rvs = func.rvs(b, scale = c, loc = a, size = 1000)
        args = (b,a,c)
        print('kstest: ',stats.kstest(xx,func.name,args ))
        print(
                    {
                        'shape':b,
                        'loc':a,
                        'scale':c
                    }
                )

        return b, rvs[np.where(rvs>=min(xx))]

    def fit_pareto(self,xx,yy):
        '''
        A method to fit the parameters for a powerlaw distribution of  density
        '''
        b,a,c = stats.pareto.fit(xx)
        rvs = stats.pareto.rvs(b, scale = c, loc = a, size = 1000)
        args = (b,a,c)
        print('kstest: ',stats.kstest(xx,'pareto',args ))
        

        return b, rvs

   


## run 
a = Analysis()

