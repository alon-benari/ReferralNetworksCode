import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import json
import pandas as pd
from os import getcwd
import re

import pyodbc

from matplotlib import pyplot as plt

from Utils.ORD_Static import Static 




class Consumption(object):
    '''
    A set of method to study the consumption of referal services in a given sta3n in a manner not different fron Request130
    that dealt with community Care
    '''
    def __init__(self, db, sta3n):
        '''
        A method to set a network
        '''
       
        self.sta3n = sta3n 
        s = Static(db)
        sql = s.get_sql(sta3n)
        self.data = s.get_data(sta3n = sta3n)
        self.fips = self.data.FIPS.unique() # abstract the count of 
        self.stopcode2name = {r[1][0]:r[1][1] for r in self.data[['VisitAtPrimaryStopCode', 'VisitAtPrimaryStopCodeName']].drop_duplicates().iterrows()}
        self.rank = 20 # number of serivces to ranks
    
    def cc_fips(self, fips):
        '''
        A method to return a dict keyed by fips and the ranked StopCode used in that fips
        '''
        fips_dict = {
                fips:self.data[['FIPS','VisitAtPrimaryStopCode']].query("FIPS =='"+fips+"'")\
                        .groupby('VisitAtPrimaryStopCode').count()['FIPS']\
                        .sort_values(ascending = False).iloc[:self.rank]
                }
        return {self.stopcode2name[k]:v for k,v in fips_dict[fips].items()}        

    def get_all_fips(self):
        '''
        
        a method to return all fips in a given sta3n and their referal of services summary
        '''
        return {f: self.cc_fips(f) for f in self.fips}
    
cc = Consumption('SQL33', 612)