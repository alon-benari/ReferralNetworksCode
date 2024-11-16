from d3blocks import D3Blocks
import pandas as pd
import Utils.ORD_ConsultUtils as u
from matplotlib import pyplot as plt, animation
import networkx as nx
import random
import numpy as np

class Visual:
    '''
    A set of method to render visualization for the project.
    '''
    def __init__(self):
        '''
        Instantiate the class
        '''
        self.d3 = D3Blocks()
        utils = u.ConsultUtils()
        self.graph_sta3n = utils.load_graph_iter()

    def load_graph(self, g):
        '''
        A method to load a graph into the class
        '''
        self.G = g
        es = lambda x,y : g.edges()[(x,y)]
        self.df = pd.DataFrame([[es(e0,e1)['sourceName'],es(e0,e1)['targetName'], es(e0,e1)['count']] for (e0,e1) in g.edges()],
            columns = ['source','target','weight'])


    def chord(self):
        '''
        Render a Chord graph  from data 
        '''
        self.d3.chord(self.df)

    def chord(self, title):
        '''
        plot a circualr chord graph for a given station

        '''
        #es = lambda x,y : g.edges()[(x,y)]
        #df = pd.DataFrame([[es(e0,e1)['sourceName'],es(e0,e1)['targetName'],
        #es(e0,e1)['count']] for (e0,e1) in g.edges()], columns =
        #['source','target','weight'])
        self.d3.chord(self.df, filepath = './notes/chord' + title + '.html',)
        
    def graph(self, title):
        '''
        plot a graph for a given networks
        '''
        self.d3.d3graph(self.df, filepath = './notes/chord' + title + '.html',)

    def demo_network_TEDx(self, N):
        '''
        A method to  demo a baby network, for the TEDx talk
        '''
        plt.rcParams["figure.figsize"] = [7.50, 7.50]
        plt.rcParams["figure.autolayout"] = True

        fig, (ax0) = plt.subplots(1,1)
        
        G = nx.DiGraph()

        nodes = range(N)
        G.add_nodes_from(nodes)
        for n in nodes:
            G.nodes()[n]['account'] = Account(n)

        ax0.set_title('Network Activity')
        # ax1.set_title('Amount distribution')
        # ax1.set_xlabel('amount')
        # ax1.set_ylabel('frequency')
        

        
        pos = nx.circular_layout(G)
        nx.draw_networkx(G,pos =pos, with_labels=True,ax = ax0)

        def animate(frame):
           #fig.clear()
           num1 = random.randint(0, N - 1)
           num2 = random.randint(0, N - 1)
           G.add_edges_from([(num1, num2)])
           G.nodes()[num1]['account'].reduce_amount()
           G.nodes()[num2]['account'].add_amount()

           nx.draw_networkx(G,pos = pos, with_labels=True,ax = ax0)
           G.remove_edges_from([(num1,num2)])
           data = [G.nodes()[n]['account'].amount for n in G.nodes()]
        #    ax1.set_xticks(range(np.max(data)+1))
           
        #    ax1.hist(data, bins = 2*N , color = 'b')
           
           nx.draw_networkx(G,pos = pos, with_labels=True,ax = ax0)
           
          
           
           

        ani = animation.FuncAnimation(fig, animate, frames=30, interval=250, repeat=False)

        #plt.show()
        return ani
    
    def demo_network(self,N):
        '''
        A method to demo the random edge creation for demo purposes
        '''
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

       

        fig, (ax0,ax1) = plt.subplots(1,2)
        
        G = nx.DiGraph()

        nodes = range(N)
        G.add_nodes_from(nodes)
        for n in nodes:
            G.nodes()[n]['account'] = Account(n)

        ax0.set_title('Network Activity')
        ax1.set_title('Amount distribution')
        ax1.set_xlabel('amount')
        ax1.set_ylabel('frequency')
        

        
        pos = nx.circular_layout(G)
        nx.draw_networkx(G,pos =pos, with_labels=True,ax = ax0)

        def animate(frame):
           #fig.clear()
           num1 = random.randint(0, N - 1)
           num2 = random.randint(0, N - 1)
           G.add_edges_from([(num1, num2)])
           G.nodes()[num1]['account'].reduce_amount()
           G.nodes()[num2]['account'].add_amount()

           nx.draw_networkx(G,pos = pos, with_labels=True,ax = ax0)
           G.remove_edges_from([(num1,num2)])
           data = [G.nodes()[n]['account'].amount for n in G.nodes()]
           ax1.set_xticks(range(np.max(data)+1))
           
           ax1.hist(data, bins = 2*N , color = 'b')
           
           nx.draw_networkx(G,pos = pos, with_labels=True,ax = ax0)
           
          
           
           

        ani = animation.FuncAnimation(fig, animate, frames=30, interval=250, repeat=False)

        #plt.show()
        return ani


class Account():
    '''
    A small class to hold the account of transfers
    '''
    def __init__(self, id):
        '''
        Instantiate the agent
        '''
        self.amount = 20
        self.id = id


    def check_amount(self):
        '''
        method to check if can give money
        '''
        if self.amount > 0 :
            return True
        else:
            return False
   
    def add_amount(self):
        '''
        Add to the ammount
        '''
        
        self.amount +=2

    def reduce_amount(self):
        '''
        reduce amount 
        '''
        
        self.amount -=2


v = Visual()

#a= v.demo_network(10)
# show some images of Sta3n's networks
#v.load_graph(v.graph_sta3n['612'])
#v.chord(v.df, '612')
