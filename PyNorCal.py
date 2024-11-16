import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import statsmodels.api as sm



class PowerLaw:
    '''
    A set of methods to analyze powerlaw data.
    '''

    def fit_model(self, y, df):
        '''
        Method to fit OLS  linear model
        Parameters:

        df -  data frame  of the  independent variables
        y -  dependent variable.

        returns:
        Ordinary least square object of linear model.
        '''

        X = sm.add_constant(df)
        res = sm.OLS(y,X).fit()

        return res

    def lrtest( self, ll0, ll1,df= 1):
        '''
        Perform Log-Liklihood testing
        Parameters:
       
        ll1, ll2 -  log-likelihood of the models as derived from statsmodels.api OLS model fitting method.

        return:
        Dictionary of likelihood ratio and  the p-value associated with it.

        '''

        lr = -2*(ll0- ll1)
        pval = stats.chi2.sf(lr, df)
       
        print( 'LR test {:3f} and p value {:5f}'.format(lr, pval))
        return {'likelihoodRatio':lr, 'p-val': pval}



pw = PowerLaw()