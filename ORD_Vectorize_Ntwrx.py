import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx
import Utils.ORD_ConsultUtils as u
import Utils.ORD_Static  as s
from scipy import stats
from Utils.PyNorCal import PowerLaw
from sklearn.preprocessing import normalize


class Vectorize_Ntwrx:
    '''
    A set of methods to  stack networkx as adj matrix for comparison
    in a 3D matrix
    '''
    utils = u.ConsultUtils()
    static = s.Static('ORD')
    static.get_vet_counts()
    sta3n_population = static.vet_count
    graph_sta3n = utils.load_graph_iter() 

    

    def __init__(self):
        self.sta3n_count = len(self.graph_sta3n.keys())
        self.l, self.D = self.get_stop_code_set()  # get exgaustuve set of stop codes.
        self.dict_list = {k:v for v, k in enumerate(self.l)}
        self.M = self.get_vec_list()  # dictionary sta3n:adj_matrix
        self.norm = lambda v: v/np.sqrt(v.dot(v.T))

    def get_sta3n_pair_list(self):
        '''
        return a dictionary keyed by row and column entry and their value.
        this will be used to iterate over and populate matries for looking all matrices.
        '''
        pair_dict = dict()
        for  r,first in enumerate(self.M.keys()):
            for  c,second in enumerate(self.M.keys()):
                pair_dict[(r,c)] = (first,second)
        return pair_dict

    def get_spectrum_adj_distance(self, mat1, mat2):
        '''
        A method to return the spectral distance of two  incidence adjacency matrices
        np.sqrt(np.sum(eigen1 - eigen2)^2
        from the PLoS manuscript
        '''
        binary = lambda x: (x>0).astype(int) # create a binary adjacency matrix
       
        eval1 , _ = np.linalg.eig(binary(mat1))
        eval2,_ = np.linalg.eig( binary(mat2)) 
        delta = np.asmatrix(eval1-eval2)

        return  np.sqrt(np.sum(delta.dot(delta.getH()))) # mutiply by the conjugate transpose


    def get_path_norm(self, mat1, mat2):
        '''
        A method to return the L2 normalized spectral distance of two  incidence adjacency matrices
        np.sqrt(np.sum(eigen1 - eigen2)^2
        from the PLoS manuscript
        '''

        
        mat = self.shortest_path_matrix(mat1) - self.shortest_path_matrix(mat2)
        l1 = mat.sum()
        eval,_ = np.linalg.eig((mat.T).dot(mat))
        l2= np.sqrt(eval.max())

        return l2
        



    def get_mat_norm(self, mat1, mat2 ):
        '''
        A method to return the  L2 and L1 of  difference of the adjacency matrix
        mat - a NxN  binary  matrix
        func -  a lambda function to do a matrix ransformation if needed
        '''
        
        mat = self.get_adj_mat(mat1)- self.get_adj_mat(mat2)
        l1 = mat.sum()
        eval,_ = np.linalg.eig((mat.T).dot(mat))
        l2= np.sqrt(eval.max())

        return l2


    def get_step_diff_norm(self, mat1, mat2):
         '''
         Get the d(G, G') from the  number of steps, which is another way to capture the network morphology
         mat1, mat2 are the weighted adjacncy matrices
         '''
         pass

    def get_spectrum_distance(self, mat1, mat2):
        '''
        A method to return the spectral distance of two matrices
        np.sqrt(np.sum(eigen1 - eigen2)^2
        from the PLoS manuscript
        '''
        

        #
        eval1,_ = np.linalg.eig(mat1)
        eval2,_ = np.linalg.eig( mat2) 
        delta = np.asmatrix(eval1-eval2)

        return  np.sqrt(np.sum(delta.dot(delta.getH()))) # mutiply by the conjugate transpose

    def compute_dot_product(self,mat1,mat2):
        '''
        This method computes the dot product of normalized incidence matrices and returns the diagonal
        '''
        mat1 = normalize(mat1, norm = 'l2', axis = 1)
        mat2 = normalize(mat2, norm = 'l2', axis = 1)
        return np.diag(mat1.dot(mat2)).mean()


    def get_diff_laplacian(self):
        '''
        A method to return the spectral distance between networks usin the eigen values of the Laplacian
        '''
        n = len(self.get_vec_list().keys())
        diff_mat = np.zeros((n,n) , dtype=complex)
        M = self.M
        pair_dict = self.get_sta3n_pair_list()

        for (r,c),(first,second) in self.get_sta3n_pair_list().items():
            mat1 = nx.from_numpy_matrix(self.M[str(first)], create_using  = nx.DiGraph)
            mat2 = nx.from_numpy_matrix(self.M[str(second)], create_using = nx.DiGraph)
            
            #
            L1 = nx.directed_laplacian_matrix(mat1, weight = 'weight')
            L2 = nx.directed_laplacian_matrix(mat2, weight = 'weight')
            #
            eVal1, _ = np.linalg.eig(L1)
            eVal2, _ = np.linalg.eig(L2)
            #
            delta = eVal1 - eVal2
            diff_mat[r,c] = np.sqrt(np.dot(delta, delta.T))
        return diff_mat

        
    def get_simple_diff(self, mat1, mat2):
        '''
        A method to return the dot product of mat1 and mat2
        mat1, mat2  -  input matrices

        '''

        
        N, M = mat1.shape
        mat1 = self.norm(mat1.reshape(1,N*M))
        mat2 = self.norm(mat2.reshape(1,N*M))

        return np.dot(mat1, mat2.T)

        
             
    
    def get_diff_matrix(self, axis, norm, normalize_flag , func):
        '''
        depricate
        A method to return the matrix of differene func for summary and visualization
        func -   is th method to compute the metric.
        axis -  axis along which to perform normalization
        norm -  type of normalization 
        '''
       
        diff_mat = np.zeros((self.sta3n_count,self.sta3n_count), dtype=complex) # A NxN matrix for all sta3ns
        M = self.get_vec_list()
        #
        
        pair_dict = self.get_sta3n_pair_list()
        for (r,c),(first,second) in self.get_sta3n_pair_list().items():
            if normalize_flag:
                mat1 = normalize(M[str(first)], axis = axis, norm = norm)
                mat2 = normalize(M[str(second)], axis = axis, norm = norm)
            else:
                mat1 = M[str(first)]
                mat2 = M[str(second)]

            #
            
            diff_mat[r,c] = func(mat1, mat2)

        return diff_mat



    #def normalize_matrix(self, mat):
    #    '''
    #    return a  normalized matrix across rows
    #    mat - raw adj matrix
    #    '''
    #    N, M  = mat.shape
        
    #    norm_mat = np.array([self.norm(mat[n,:]) for n in range(M)])
    #    return np.nan_to_num(norm_mat, 0)
        
    def get_upper_trig(self, mat):
        '''
        A method to return the upper triangle of a matrix with its diaonal rendered to 0s

        mat -  a matrix to be processed.
        '''
        upper = np.triu(mat) - np.diag(np.diag(mat))

        return {
        'upper' : upper,
        'data': upper[upper!=0],
        'mean' : upper[upper!=0].mean(),
        'std' : upper[upper!=0].std(),
        'percentile':np.percentile(upper[upper!=0], [5,25,50,75,95])
    }

    def get_vec_list(self):
        '''
        Return a dictionary of adjacnecy matrices
        '''
        sta3ns= {k: v['vet_count'] for k,v  in self.static.vet_count.items()} # set of sta3nd and population
        sorted_sta3ns = [sta for sta, pop in sorted(sta3ns.items(), key = lambda item: item[1], reverse = True)]
        
        sorted_sta3ns = [n for n in sorted_sta3ns if len(n) == 3]
        return {k: self.vectorize_ntwrx(k) for k in sorted_sta3ns}  # return a sorted list of matrics



    def vectorize_ntwrx(self,sta3n):
        '''
        A method to return a 3 D matrix (ajd matrix ) of all the referral networks
        '''
        #l, D = self.get_stop_code_set()
        dict_list = {k:v for v, k in enumerate(self.l)} # return stop_code (key) column # (value) pair.
        M = np.zeros((self.D,self.D))  # set a matrix
        G = self.graph_sta3n[sta3n]
        for (s,t) in G.edges():
            M[dict_list[s],dict_list[t]] = G.edges()[(s,t)]['count'] # weighted adj matrix 

        return M

    def get_va_wide_norm(self, sc, direction ):
        '''
        return a normed matrix with outgoing/in going edges from a given sta3nacross all sta3ns
        '''
        stop_code =  self.dict_list[sc]
        if (direction == 'out'):
            M  = np.vstack([self.norm(m[stop_code,:]) for m in self.M.values()])
        if (direction == 'in'):
            M  = np.vstack([self.norm(m[:,stop_code]) for m in self.M.values()])
        M =np.nan_to_num(M, nan = 0.0) 
        A = M.dot(M.T)
        diagonal = np.diag(A)


        return np.triu(A) - np.diag(diagonal)
            

        


    def scale_edge_stop_code(self, sc, direction):
        '''
        A method to  return for a given  stop code count  of outgoing/incoming requests across all stations.
        '''
        id =  self.dict_list[sc]

        xy = []

        if (direction == 'out'):
           d = {k:v[id,:].sum()  for k,v in self.M.items()}

        if (direction == 'in'):
           d = {k:v[:,id].sum() for k,v in self.M.items()}

        
        for k, v in self.sta3n_population.items():
            try:
                xy.append([v['vet_count'], d[k[:3]]])
                

            except Exception as e:
                print(e)


        return np.array(xy)[:,0], np.array(xy)[:,1]
    
    



   

    def get_stop_code_set(self):
        '''
        Return a set of stop code set
        '''
        node_set = lambda gr: list(gr.nodes) # return a set of nodes for graph gr.
        node_list = []
        for gr in self.graph_sta3n.values():
            node_list.extend(node_set(gr))
        return  set(node_list),len(set(node_list))
                
    def shortest_path_matrix(self, mat):
        '''
        Return a matrix of shortest paths between nodes.
        The matrix mat is converte to an incidence matrix, and then computed
        mat - a weighted adjacency matrix
        '''
        mat = self.get_adj_mat(mat) # binary adjacnecy matrix
        N,M = mat.shape
        path_mat = np.zeros((N,M))
        #
        g = nx.from_numpy_matrix(mat, create_using = nx.DiGraph)
        path_length = np.array([[ e0,e1,len(nx.shortest_path(g,e0,e1))] for (e0,e1)   in  g.edges()])
        path_mat[path_length[:,0],path_length[:,1]] = path_length[:,2] # create path matrix

        return path_mat - np.diag(np.diag(path_mat)) # path matrix


    def get_adj_mat(self, mat):
        '''
        return the incidence Adjacency matrix  of mat
        '''
        
        return (mat>0).astype(int)



vn = Vectorize_Ntwrx()
#vn.get_vec_list   # return a dictionary of sta3n: wt adj matrix
#N = 
#N = vn.get_va_wide_norm(315,'in') return the normal matrix of in degre to Stop code 315
#N.dot(N.T) dot product but since this is normalized, this is the cosine between -1 and 1.