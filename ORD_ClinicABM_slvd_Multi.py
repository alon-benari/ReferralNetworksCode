import agentpy as ap
import networkx as nx
import random
import scipy.stats as stats
import numpy as np
import Utils.ORD_ConsultUtils as u

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython

'''
A set of methods to set a ABM for a random graph and beyond
'''
class ClinicAgent(ap.Agent):
    '''
    A method  to define a clinic agent for am ABM model
    '''
    
    
    def setup(self):
        '''
        Instantiate a clinic agent
        '''
        #self.send_consult = stats.bernoulli.rvs(p =self.consult_probability )
        self.weibull_dict = dict()
        self.condition = 0
        self.queue = list() # a list to hold the consults and the time to their maturity
        self.state = 0 # flag to show if a clinic is over the CITC cutoff.
       

    def add_to_queue(self):
        '''
        A method to add a time stamp to queue
        '''
        #self.queue.append(stats.expon.rvs(scale = 1/self.tau, loc = 0,size = 1) + self.model.t)
        self.queue.append(stats.exponweib.rvs(*self.weibull ,size = 1) + self.model.t)
        self.queue.append(stats.exponweib.rvs(*self.weibull ,size = 1) + self.model.t)
       
        
    def update_queue(self):
        '''
        remove entries from whom the time has come.
        '''
        
        self.queue = list(filter(lambda x: x>self.model.t, self.queue))
        


    def consult_probability(self):
        '''
        Set the probability of sending a consult
        '''
        
        return stats.bernoulli.rvs(self.prob)
  
    def update_weibull(self):
        '''
        A method to update the time constant coming from weibull
        '''
        print(self)
        print('Just Update weibull :{0}'.format(self))
        params = list(self.weibull)
        params[2] = params[2]* self.p.shape_scaler
        self.weibull = tuple(params)
        self.weibull_dict[self.model.t] = tuple(params)
        print({self:params})

    def choose_service(self):
        '''
        select a service to send a consult to
        '''
        data = self.choice_density
        #return 

       
        return [self.model.agent2clinic[c] for c in random.choices(data['choice'], data['density'],\
                                                                #k = stats.poisson(mu = data['poisson']).rvs(size = 1)[0])]
                                                                  k= stats.binom(n = 365, p = self.prob).rvs(1)[0])]
                            
    def consult_num(self):
        '''
        A method to return the number of consults a service fires at a given time.
        Treat this as a possion distribution.  Where the mean number of consults devided by the number of steps
        '''
        pass

    
    def submit_consult(self):
        '''
        pefrom the submission of a consult
        '''
        self.update_queue()
        if stats.bernoulli.rvs(self.prob):
            target_clinic = self.choose_service() 
            for tc in target_clinic:
                tc.add_to_queue()
                if (len(tc.queue) >= self.model.p.CITC):
                    tc.condition = 1
                    if self.model.t >=self.p.weibull_flag:
                        tc.update_weibull()
                   
                else:
                    tc.condition = 0




class ClinicModel(ap.Model):
    '''
    A class to run the simulation

    '''

    def set_node_prob(self, node):
        '''
        set the prob of a node will fire  consults,
        set the density of probabilities to which service it will consult
        node - target node
        '''
        gr = self.network.graph
        density = {target: gr.edges()[(node,target)]['count'] for target  in gr.successors(node)}

        gr.nodes()[node]['prob_estim']=np.sum([v for v in density.values()])/float(self.p.sum_consult)
   
        gr.nodes()[node]['choice_density'] = {
                                                'choice':list(density.keys()), 'density':np.array(list(density.values()))/np.sum(list(density.values()))
                                            }
        return {
            'prob':np.sum([v for v in density.values()])/float(self.p.sum_consult),
            'choice_density':{
                                'choice':list(density.keys()),
                                'density':np.array(list(density.values()))/np.sum(list(density.values())),
            'poisson': np.mean([v for v in density.values()])/float(self.p.real_steps)
                                            }
            }
       

    def setup(self):
        '''
        initialize agens and network for the model
        '''
        
        
        graph = self.p.graph
        self.agent2clinic = {}
        

        # create agens and network 
        self.agents =  ap.AgentList(self,self.p.population, ClinicAgent)

        self.network = self.agents.network =  ap.Network(self,graph = graph)
        gr = self.network.graph
        self.network.add_agents(self.agents, self.network.nodes)
        # load data from a graph
        for a,n in zip(self.network.agents, gr.nodes()):
            self.agent2clinic[n] = a
          
            #a.tau =  np.abs(gr.nodes()[n]['tau']) # time constant
            
            a.weibull = gr.nodes()[n]['weibull']
            a.weibull_dict[0] = gr.nodes()[n]['weibull']
            #
            a.children = list(self.network.graph.successors(n))
            #
            probs = self.set_node_prob(n)
            a.prob = probs['prob'] # prob of firing a consult
            a.choice_density = probs['choice_density'] #data to choose a clinic to conuslt
            #

        


        

    #
    def update(self):
        '''
        Record variables after setup for each step
        '''
        total_consults = sum([len(a.queue) for a in self.model.agents])
       
        mean_consults = np.mean([len(a.queue) for a in self.model.agents])
        sum_CITC = np.sum([a.condition for a in self.model.agents])
        mean_CITC = np.mean([a.condition for a in self.model.agents])
        #
        self.record('mean_of_q', mean_consults)
        self.record('current_in_q',total_consults)
        self.record('CITC', sum_CITC)
        

            
 
           
            
    
    def step(self):
        '''
        eacg agent goes through this.
        '''
       
        self.agents.submit_consult()
            

    def end(self):
        '''
        Record measures
        '''
        #self.report("The total of Z:",self.Zero)
        #self.report("The total of O:",self.One)



    
##run

random.seed(123)
utils = u.ConsultUtils()
graph_sta3n = utils.load_graph_iter() # load a bunch of graphs as a dictionary 

#
# create some dummy data.  
# 


def cartoon(model, axs):
    '''
    A method to show a visualization of the simulation in acation
    '''
    it = lambda x: [a for a in x]

    ax0, ax1 = axs
    queued = [ len(a.queue) for a in it(model.agents)]
    ax0.hist(queued, bins = 10)# np.max(queued))
    #
    
    col_dict =  { 0:'b', 1:'r'}
    color_conditions = [col_dict[c] for c in [ b.condition for b in it(model.agents)]]

    nx.draw_circular(
            model.network.graph, node_color = color_conditions, node_size = 25, ax = ax1
            )


def abm_random():
    '''
    data and methods to run a simulation on a  random graph, random time constants etc for demo purposes
    '''

    graph = nx.erdos_renyi_graph(100,0.12, seed = 123,
                            directed = True)

    for (es0,es1) in graph.edges():
        graph.edges()[(es0,es1)]['count'] = random.randint(2,110) # set weights fo number of consults

    # sum of all consults
    sum_consult =  np.sum([graph.edges()[(e0,e1)]['count'] for e0,e1 in graph.edges()])

    for n in graph.nodes():
            
            graph.nodes()[n]['tau']= random.randint(30,90)
            graph.nodes()[n]['queue'] = list()
                
    parameters = {

    'graph' : graph,
    'population' : len(graph.nodes()),
    'steps':500,
    
    'CITC_cutoff':10,
    'sum_consult':np.sum([graph.edges()[(e0,e1)]['count'] for e0,e1 in graph.edges()]),#sum_consult
    'real_steps': 365 # number of day in a year for which the data exists
    

    }
    return parameters
#
# use a real world network
###

def sta3n_params(sta3n):
    '''
    Return a params for a set graph for a give sta3n
    sta3n - string  for the station number
    '''
    graph = graph_sta3n[sta3n]
    for n in graph.nodes():

        #graph.nodes()[n]['tau'] = np.mean(
        #                                    np.abs(graph.nodes()[n]['t2appt'])
        #                                    )
        params = stats.exponweib.fit(np.abs(graph.nodes()[n]['t2appt']), floc = 0,  f0=1)
        graph.nodes()[n]['weibull'] = params #{'shape': params[1],'scale':params[3]}
        

        
    return {
    'graph' : graph,
    'population':len(graph.nodes()),
    'steps':300,
    'weibull_flag':150,
    'CITC':10,
    'sum_consult':np.sum([graph.edges()[(e0,e1)]['count'] for e0,e1 in graph.edges()]),#sum_consult
    'shape_scaler': ap.Range(0.8,1.0), # 10% incease in weibull shape coeffient
    'real_steps': 365 # number of day in a year for which the data exists
    

    }



#parameters = abm_random() # use for random graphs
random.seed(123)
utils = u.ConsultUtils()
graph_sta3n = utils.load_graph_iter() # load a bunch of graphs as a dictionary 

parameters = sta3n_params('612')
'''
model = ClinicModel(parameters)
results = model.run()
fig, (ax0,ax1,ax2,ax3) = plt.subplots(4,1, figsize = (14,10))
results.variables.ClinicModel['mean_of_q'].plot(ax = ax0)
ax0.set_ylabel('mean')

results.variables.ClinicModel['CITC'].plot(ax = ax1)
ax1.set_ylabel('#CITC')

results.variables.ClinicModel['current_in_q'].plot(ax = ax2)

ax2.set_ylabel('currnet_in_q')

results.variables.ClinicModel['current_in_q'].hist(ax = ax3)
ax3.set_ylabel('dist_current_in_queue')
plt.show()
'''


sample = ap.Sample(
    parameters,
    n = 2 ,
    method='saltelli',
    calc_second_order=False
)
exp = ap.Experiment(ClinicModel, sample, record = True,iterations = 5)
results = exp.run()






