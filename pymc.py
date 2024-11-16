"""
Inferring a binomial proportion using PyMC.
"""
from multiprocessing.dummy import freeze_support
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import numpy as np
import pymc3 as pm
import scipy.stats as stats
import theano.tensor as tt


def main():
    # Generate the data
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])  # 11 heads and 3 tails


    with pm.Model() as model:
        # define the prior
        theta = pm.Beta('theta', 1., 1.)  # prior
        # define the likelihood
        y = pm.Bernoulli('y', p=theta, observed=y)

        # Generate a MCMC chain
        trace = pm.sample(1000)


    # create an array with the posterior sample
    theta_sample = trace['theta']
    return theta_sample
"""
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(theta_sample[:500], np.arange(500), marker='o', color='skyblue')
    ax[0].set_xlim(0, 1)
    ax[0].set_xlabel(r'$\theta$')
    ax[0].set_ylabel('Position in Chain')

    pm.plot_posterior(theta_sample, ax=ax[1], color='skyblue');
    ax[1].set_xlabel(r'$\theta$');

    # Posterior prediction:
    # For each step in the chain, use posterior theta to flip a coin:
    y_pred = np.zeros(len(theta_sample))
    for i, p_head in enumerate(theta_sample):
        y_pred[i] = np.random.choice([0, 1], p=[1 - p_head, p_head])

    # Jitter the 0,1 y values for plotting purposes:
    y_pred_jittered = y_pred + np.random.uniform(-.05, .05, size=len(theta_sample))

    # Now plot the jittered values:
    plt.figure()
    plt.plot(theta_sample[:500], y_pred_jittered[:500], 'C1o')
    plt.xlim(-.1, 1.1)
    plt.ylim(-.1, 1.1)
    plt.xlabel(r'$\theta$')
    plt.ylabel('y (jittered)')

    mean_y = np.mean(y_pred)
    mean_theta = np.mean(theta_sample)

    plt.plot(mean_y, mean_theta, 'k+', markersize=15)
    plt.annotate('mean(y) = %.2f\nmean($\\theta$) = %.2f' %
        (mean_y, mean_theta), xy=(mean_y, mean_theta))
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.savefig('BernBetaPyMCPost.png')
    plt.show()
"""

def main2():

    # The parameters are the bounds of the Uniform.
    
    p_true = 0.05  # remember, this is unknown.
    N = 1500

    occurrences = stats.bernoulli.rvs(p_true, size=N)

    print(occurrences) # Remember: Python treats True == 1, and False == 0
    print(np.sum(occurrences))
    with pm.Model() as model:
        p = pm.Uniform('p', lower=0, upper=1)
    #include the observations, which are Bernoulli
    with model:
        obs = pm.Bernoulli("obs", p, observed=occurrences)
        # To be explained in chapter 3
        step = pm.Metropolis()
        trace = pm.sample(18000, step=step)
        burned_trace = trace[1000:]
    return burned_trace
    
    



def main3():
    true_p_A = 0.05
    true_p_B = 0.04

    #notice the unequal sample sizes -- no problem in Bayesian analysis.
    N_A = 1500
    N_B = 750

    #generate some observations
    observations_A = stats.bernoulli.rvs(true_p_A, size=N_A)
    observations_B = stats.bernoulli.rvs(true_p_B, size=N_B)
    print("Obs from Site A: ", observations_A[:30], "...")
    print("Obs from Site B: ", observations_B[:30], "...")

    with pm.Model() as model:
        p_A = pm.Uniform("p_A", 0, 1)
        p_B = pm.Uniform("p_B", 0, 1)
        
        # Define the deterministic delta function. This is our unknown of interest.
        delta = pm.Deterministic("delta", p_A - p_B)

        
        # Set of observations, in this case we have two observation datasets.
        obs_A = pm.Bernoulli("obs_A", p_A, observed=observations_A)
        obs_B = pm.Bernoulli("obs_B", p_B, observed=observations_B)

        # To be explained in chapter 3.
        step = pm.Metropolis()
        trace = pm.sample(20000, step=step)
        burned_trace=trace[1000:]

        p_A_samples = burned_trace["p_A"]
        p_B_samples = burned_trace["p_B"]
        delta_samples = burned_trace["delta"]

        return delta_samples

def main4():
    N = 100
    X = 35
    with pm.Model() as model:
        p = pm.Uniform("freq_cheating",0,1)
        true_answers = pm.Bernoulli("truths", p , shape = N, testval = np.random.binomial(1,0.5,N))
        first_coin_flips = pm.Bernoulli("first_flips", p=0.5 , shape = N, testval = np.random.binomial(1,0.5,N))
        second_coin_flips = pm.Bernoulli("second_flips", 0.5, shape=N, testval=np.random.binomial(1, 0.5, N))
        val = first_coin_flips* true_answers +(1-first_coin_flips)*second_coin_flips
        observed_prop = pm.Deterministic("observed_prop", tt.sum(val)/float(N))
        observations = pm.Binomial("obs",N, observed_prop,observed = X)
    with model:
        step = pm.Metropolis(vars=[p])
        trace = pm.sample(40000, step=step)
        burned_trace = trace[15000:]

    return burned_trace


if __name__ == "__name__":
    freeze_support()
#     a = main()
#a = main()

