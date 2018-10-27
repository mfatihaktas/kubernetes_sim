import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plot

import math, random, numpy, scipy
import numpy as np
from scipy.stats import *

from plot_utils import *
from log_utils import *

class RV(): # Random Variable
  def __init__(self, l_l, u_l):
    self.l_l = l_l
    self.u_l = u_l

class Normal(RV):
  def __init__(self, mu, sigma):
    RV.__init__(self, l_l=float('-Inf'), u_l=float('Inf') )
    self.mu = mu
    self.sigma = sigma
    
    self.dist = scipy.stats.norm(mu, sigma)
  
  def __repr__(self):
    return 'Normal[mu= {}, sigma= {}]'.format(self.mu, self.sigma)
  
  def cdf(self, x):
    return self.dist.cdf(x)
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def mean(self):
    return self.mu
  
  def sample(self):
    return self.dist.rvs(size=1)[0]

class TNormal(RV):
  def __init__(self, mu, sigma, max_value=np.inf):
    RV.__init__(self, l_l=0, u_l=max_value)
    sigma = sigma if sigma != 0 else 0.01
    self.mu = mu
    self.sigma = sigma
    
    lower, upper = 0, max_value
    a = (lower - mu)/sigma
    b = (upper - mu)/sigma
    self.dist = scipy.stats.truncnorm(a, b, loc=mu, scale=sigma)
    # mean, var, skew, kurt = scipy.stats.truncnorm.stats(a, b, moments='mvsk')
    # blog(mean=mean, var=var, skew=skew, kurt=kurt)
  
  def __repr__(self):
    return 'TNormal[mu= {}, sigma= {}]'.format(self.mu, self.sigma)
  
  def cdf(self, x):
    return self.dist.cdf(x)
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def mean(self):
    return self.dist.mean()
  
  def std(self):
    return self.dist.std()
  
  def sample(self):
    return self.dist.rvs(size=1)[0]

class Exp(RV):
  def __init__(self, mu, D=0):
    RV.__init__(self, l_l=D, u_l=np.inf)
    self.D = D
    self.mu = mu
  
  def __repr__(self):
    if self.D == 0:
      return r'Exp(\mu={})'.format(self.mu)
    return r'{} + Exp(\mu={})'.format(self.D, self.mu)
  
  def tail(self, x):
    if x <= self.l_l:
      return 1
    return math.exp(-self.mu*(x - self.D) )
  
  def cdf(self, x):
    if x <= self.l_l:
      return 0
    return 1 - math.exp(-self.mu*(x - self.D) )
  
  def pdf(self, x):
    if x <= self.l_l:
      return 0
    return self.mu*math.exp(-self.mu*(x - self.D) )
  
  def mean(self):
    return self.D + 1/self.mu
  
  def var(self):
    return 1/self.mu**2
  
  def moment(self, i):
    return moment_ith(i, self)
  
  def laplace(self, s):
    if self.D > 0:
      log(ERROR, "D= {} != 0".format(D) )
    return self.mu/(s + self.mu)
  
  def sample(self):
    return self.D + random.expovariate(self.mu)

class HyperExp(RV):
  def __init__(self, p_l, mu_l):
    RV.__init__(self, l_l=0, u_l=np.inf)
    self.p_l = p_l
    self.mu_l = mu_l
    
    self.X_l = [Exp(mu) for mu in mu_l]
    
    self.i_l = [i for i in range(len(self.p_l) ) ]
    self.dist_for_gensample = scipy.stats.rv_discrete(name='hyperexp', values=(self.i_l, self.p_l) )
  
  def __repr__(self):
    return r'HyperExp(p= {}, \mu= {})'.format(self.p_l, self.mu_l)
  
  def pdf(self, x):
    return sum([self.p_l[i]*X.pdf(x) for i, X in enumerate(self.X_l) ] )
  
  def cdf(self, x):
    return sum([self.p_l[i]*X.cdf(x) for i, X in enumerate(self.X_l) ] )
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def mean(self):
    return sum([self.p_l[i]*X.mean() for i, X in enumerate(self.X_l) ] )
  
  def moment(self, m):
    return sum([self.p_l[i]*X.moment(m) for i, X in enumerate(self.X_l) ] )
  
  def laplace(self, s):
    return sum([self.p_l[i]*X.laplace(s) for i, X in enumerate(self.X_l) ] )
  
  def sample(self):
    i = self.dist_for_gensample.rvs()
    return self.X_l[i].sample()

class Pareto(RV):
  def __init__(self, loc, a):
    RV.__init__(self, l_l=loc, u_l=np.inf)
    self.loc = loc
    self.a = a
  
  def __repr__(self):
    return "Pareto(loc= {}, a= {})".format(self.loc, self.a)
  
  def tolatex(self):
    return r'Pareto(\min= {}, \alpha= {})'.format(self.loc, self.a)
  
  def tail(self, x):
    if x <= self.l_l:
      return 1
    return (self.loc/x)**self.a
  
  def cdf(self, x):
    if x <= self.l_l:
      return 0
    return 1 - (self.loc/x)**self.a
  
  def pdf(self, x):
    if x <= self.l_l:
      return 0
    return self.a*self.loc**self.a / x**(self.a+1)
  
  def dpdf_dx(self, x):
    if x <= self.l_l:
      return 0
    return sympy.mpmath.diff(lambda y: self.a*self.loc**self.a / y**(self.a+1), x)
  
  def mean(self):
    if self.a <= 1:
      log(WARNING, "Mean is Infinity; a= {} <= 1".format(self.a) )
      return np.inf
    else:
      return self.loc*self.a/(self.a-1)
  
  def var(self):
    if self.a <= 2:
      log(WARNING, "Variance is Infinity; a= {} <= 2".format(self.a) )
      return np.inf
    else:
      return self.a*self.loc**2 / (self.a-1)**2/(self.a-2)
  
  def sample(self):
    return ((numpy.random.pareto(self.a, 1) + 1)*self.loc)[0]
    # return pareto.ppf(numpy.random.uniform(0, 1), b=self.a, scale=self.loc)

class TPareto(): # Truncated
  def __init__(self, l, u, a):
    RV.__init__(self, l_l=l, u_l=u)
    self.l = l
    self.u = u
    self.a = a
  
  def __repr__(self):
    return "TPareto(l= {}, u= {}, a= {})".format(self.l, self.u, self.a)
  
  def tolatex(self):
    return r'TPareto(\min= {}, \max= {}, \alpha= {})'.format(self.l, self.u, self.a)
  
  def pdf(self, x):
    if x < self.l: return 0
    elif x >= self.u: return 0
    else:
      return self.a*self.l**self.a * 1/x**(self.a+1) / (1 - (self.l/self.u)**self.a)
  
  def cdf(self, x):
    if x < self.l: return 0
    elif x >= self.u: return 1
    else:
      return (1 - (self.l/x)**self.a)/(1 - (self.l/self.u)**self.a)
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def mean(self):
    return self.moment(1)
  
  def std(self):
    return math.sqrt(self.moment(2) - self.mean()**2)
  
  def moment(self, k):
    if k == self.a:
      return math.log(self.u_l/self.l)
    else:
      return self.a*self.l**k/(self.a-k) * \
             (1 - (self.l/self.u)**(self.a-k))/(1 - (self.l/self.u)**self.a)
  
  def sample(self):
    r = random.uniform(0, 1)
    s = self.l*(1 - r*(1-(self.l/self.u)**self.a) )**(-1/self.a)
    if s < self.l or s > self.u:
      log(ERROR, "illegal sample! s= {}".format(s) )
      return None
    return s

class SimRV(RV):
  def __init__(self, sample_l):
    RV.__init__(self, l_l=min(sample_l), u_l=max(sample_l) )
    
    self.sample_l = sample_l
    self.num_sample = len(self.sample_l)
  
  def __repr__(self):
    return "SimRV"
  
  def mean(self):
    return sum(self.sample_l)/self.num_sample
  
  def sample(self):
    return self.sample_l[math.floor(self.num_sample*random.random() ) ]

class Dolly(RV):
  # Kristen et al. A Better Model for Job Redundancy: Decoupling Server Slowdown and Job Size
  def __init__(self):
    RV.__init__(self, l_l=1, u_l=12)
    
    self.v = numpy.arange(1, 13)
    self.p = [0.23, 0.14, 0.09, 0.03, 0.08, 0.1, 0.04, 0.14, 0.12, 0.021, 0.007, 0.002]
    self.dist = scipy.stats.rv_discrete(name='dolly', values=(self.v, self.p) )
  
  def __repr__(self):
    return "Dolly[{}, {}]".format(self.l_l, self.u_l)
  
  def tolatex(self):
    return "Dolly"
  
  def pdf(self, x):
    return self.dist.pmf(x) if (x >= self.l_l and x <= self.u_l) else 0
  
  def cdf(self, x):
    if x < self.l_l:
      return 0
    elif x > self.u_l:
      return 1
    return float(self.dist.cdf(math.floor(x) ) )
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def mean(self):
    return self.moment(1)
  
  def moment(self, m):
    return sum([self.v[i]**m*self.p[i] for i in range(12) ] )
  
  def sample(self):
    return random.randint(self.l_l, self.u_l)
  
  def sample(self):
    u = random.uniform(0, 1)
    return self.dist.rvs()[0] + u/100

class Bern(RV):
  def __init__(self, L, U, p):
    RV.__init__(self, l_l=L, u_l=U)
    self.p = p
    
    self.v_l = [L, U]
    self.p_l = [1 - p, p]
    self.dist = scipy.stats.rv_discrete(name='bern', values=(self.v_l, self.p_l) )
  
  def __repr__(self):
    return "Bern(l= {}, u= {}, p= {})".format(self.l_l, self.u_l, self.p)
  
  def pdf(self, x):
    return self.dist.pmf(x)
  
  def cdf(self, x):
    if x < self.l_l:
      return 0
    elif x > self.u_l:
      return 1
    return self.dist.cdf(math.floor(x) )
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def mean(self):
    return self.dist.mean()
  
  def moment(self, i):
    return self.dist.moment(i)
  
  def sample(self):
    return self.dist.rvs()[0]

class DUniform():
  def __init__(self, lb, ub):
    RV.__init__(self, l_l=lb, u_l=ub)
    
    self.v = numpy.arange(self.l_l, self.u_l+1)
    w_l = [1 for v in self.v]
    self.p = [w/sum(w_l) for w in w_l]
    self.dist = scipy.stats.rv_discrete(name='duniform', values=(self.v, self.p) )
  
  def __repr__(self):
    return "DUniform[{}, {}]".format(self.l_l, self.u_l)
  
  def mean(self):
    return (self.u_l + self.l_l)/2
  
  def pdf(self, x):
    return self.dist.pmf(x)
  
  def cdf(self, x):
    if x < self.l_l:
      return 0
    elif x > self.u_l:
      return 1
    return self.dist.cdf(math.floor(x) )
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def moment(self, i):
    # p = 1/(self.u_l - self.l_l + 1)
    # return sum([p*v**i for v in range(self.l_l, self.u_l+1) ] )
    return self.dist.moment(i)
  
  def sample(self):
    # return random.randint(self.l_l, self.u_l)
    return self.dist.rvs()[0]

class BZipf():
  def __init__(self, lb, ub, a=1):
    RV.__init__(self, l_l=lb, u_l=ub)
    self.a = a
    
    self.v = numpy.arange(self.l_l, self.u_l+1) # values
    w_l = [float(v)**(-a) for v in self.v] # self.v**(-a) # weights
    self.p = [w/sum(w_l) for w in w_l]
    self.dist = scipy.stats.rv_discrete(name='bounded_zipf', values=(self.v, self.p) )
  
  def __repr__(self):
    return "BZipf([{}, {}], a= {})".format(self.l_l, self.u_l, self.a)
  
  def pdf(self, x):
    return self.dist.pmf(x)
  
  def cdf(self, x):
    # if x < self.l_l: return 0
    # elif x >= self.u_l: return 1
    # else:
    #   return sum(self.p[:(x-self.l_l+1) ] )
    return self.dist.cdf(x)
  
  def inv_cdf(self, p):
    return self.dist.ppf(p)
  
  def tail(self, x):
    return 1 - self.cfd(x)
  
  def mean(self):
    # return sum([v*self.p(i) for i,v in enumerate(self.v) ] )
    return self.dist.mean()
  
  def sample(self):
    return self.dist.rvs(size=1)[0]

class Binom():
  def __init__(self, n, p):
    RV.__init__(self, l_l=0, u_l=n)
    self.n = n
    self.p = p
    
    self.dist = scipy.stats.nbinom(n, p)
  
  def __repr__(self):
    return "Binom[n= {}, p= {}]".format(self.n, self.p)
  
  def pdf(self, x):
    return self.dist.pdf(x)
  
  def cdf(self, x):
    return self.dist.cdf(x)
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def sample(self):
    return self.dist.rvs(size=1)[0]

class NegBinom():
  def __init__(self, num_succ, p):
    RV.__init__(self, l_l=num_succ, u_l=float('Inf') )
    self.p = p
    
    self.dist = scipy.stats.nbinom(num_succ, p)
  
  def __repr__(self):
    return "NegBinom[num_succ= {}, p= {}]".format(self.l_l, self.p)
  
  def cdf(self, x):
    return self.dist.cdf(x - self.l_l)
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def sample(self):
    return self.dist.rvs(size=1)[0]

class Gamma():
  def __init__(self, num_exp, rate):
    RV.__init__(self, l_l=0, u_l=float('Inf') )
    
    self.shape, self.scale = num_exp, 1/rate
    # self.dist = numpy.random.gamma(shape, scale, size=1)
    self.dist = scipy.stats.gamma(self.shape, self.scale)
  
  def __repr__(self):
    return "Gamma[shape= {}, scale= {}]".format(self.shape, self.scale)
  
  def cdf(self, x):
    return self.dist.cdf(x)
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def sample(self):
    return self.dist.rvs(size=1)[0]

class X_n_k():
  def __init__(self, X, n, k):
    RV.__init__(self, l_l=X.l_l, u_l=X.u_l)
    self.X, self.n, self.k = X, n, k
  
  def __repr__(self):
    return "{}_{{}:{}}".format(X, self.n, self.k)
  
  def pdf(self, x):
    return self.n*self.X.pdf(x) * binomial(self.n-1, self.k-1) * self.X.cdf(x)**(self.k-1) * self.X.tail(x)**(self.n-self.k)
  
  def cdf(self, x):
    return cdf_n_k(self.X, self.n, self.k, x)
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def moment(self, i):
    return float(mpmath.quad(lambda x: i*x**(i-1) * self.tail(x), [0, mpmath.inf] ) )
  
  def sample(self):
    return gen_orderstat_sample(self.X, self.n, self.k)

def binomial(n, k):
  return scipy.special.binom(n, k)

def moment_ith(i, X):
  # return float(mpmath.quad(lambda x: i*x**(i-1) * X.tail(x), [0, X.u_l] ) ) # mpmath.inf 10000*10
  return float(mpmath.quad(lambda x: i*x**(i-1) * X.tail(x), [0, mpmath.inf] ) )

# Order stats
def cdf_n_k(X, n, k, x): # Pr{X_n:k < x}
  cdf = 0
  for i in range(k, n+1):
    cdf += binomial(n, i) * X.cdf(x)**i * X.tail(x)**(n-i)
  return cdf

def moment_ith_n_k(X, i, n, k): # E[X_n:k]
  return mpmath.quad(lambda x: i*x**(i-1) * (1 - cdf_n_k(X, n, k, x) ), [0, 10000*10] )

def gen_orderstat_sample(X, n, k):
  # print("s_l= {}".format(s_l) )
  return sorted([X.sample() for _ in range(n) ] )[k-1]

def H(n):
  if n == 0:
    return 0
  sum_ = 0
  for i in range(1, n+1):
    sum_ += float(1/i)
  return sum_

def fact(n):
  return math.factorial(n)

def laplace(X, r):
  return mpmath.quad(lambda x: math.exp(-r*x) * X.pdf(x), [0, X.u_l] )

def distm_to_rv(dist_m):
  name = dist_m['name']
  if name == 'TPareto':
    return TPareto(dist_m['l'], dist_m['u'], dist_m['a'] )
  elif name == 'BZipf':
    return BZipf(dist_m['l'], dist_m['u'] )
  else:
    log(ERROR, "Unrecognized name= {}".format(name) )

def gen_rectangular_TNormal(mu, sigma, maxval, duration_mean):
  sigma = sigma if sigma != 0 else 0.01
  high_level = (maxval - mu)*np.random.rand() + mu
  medium_level = mu
  low_level = 2*mu - high_level
  rv_l = [TNormal(low_level, sigma, mu),
          TNormal(medium_level, sigma, high_level),
          TNormal(high_level, sigma, maxval) ]
  return RectangleRV(Exp(1/duration_mean), rv_l)

class RectangleRV():
  def __init__(self, duration_rv, rv_l):
    self.duration_rv = duration_rv
    self.rv_l = rv_l
    
    self.rv = self.rv_l[0]
    self.countdown_forchange = duration_rv.sample()
  
  def __repr__(self):
    return 'BurstyRV:\n' + \
      '  duration_rv= {}\n'.format(self.duration_rv) + \
      '  rv_l= {}\n'.format(self.rv_l)
  
  def mean(self):
    return np.mean([rv.mean() for rv in self.rv_l] )
  
  def sample(self):
    if self.countdown_forchange > 0:
      self.countdown_forchange -= 1
    else:
      self.countdown_forchange = self.duration_rv.sample()
      self.rv = np.random.choice(self.rv_l)
    return self.rv.sample()
  
class MixedRVs():
  def __init__(self, p_l, rv_l):
    self.p_l = p_l
    self.rv_l = rv_l
    self.dist_to_select_rv = scipy.stats.rv_discrete(
      name='mixed', values=(numpy.arange(0, len(p_l) ), p_l) )
  
  def __repr__(self):
    return 'MixOfRVs:\n' + \
      '  p_l= {}\n'.format(self.p_l) + \
      '  rv_l= {}\n'.format(self.rv_l)
  
  def mean(self):
    return sum([self.p_l[i]*rv.mean() for i, rv in enumerate(self.rv_l) ] )
  
  def sample(self):
    return self.rv_l[self.dist_to_select_rv.rvs() ].sample()

if __name__ == "__main__":
  # rv = TNormal(mu=1, sigma=2, max_value=10)
  # blog(rv=rv, rv_mean=rv.mean() )
  
  # '''
  rv = gen_rectangular_TNormal(mu=10, sigma=1, maxval=20, duration_mean=100)
  blog(mean=rv.mean() )
  v_l = [rv.sample() for _ in range(2000) ]
  t_l = np.arange(1, len(v_l)+1)
  plot.plot(t_l, v_l, color=NICE_ORANGE, marker='.', ls=':', lw=2, mew=3, ms=3)
  prettify(plot.gca() )
  # plot.title('{}'.format(rv) )
  # plot.legend()
  plot.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
  plot.tick_params(axis='y', which='both', left=False, top=False, labelleft=False)
  # plot.ylabel('Resource usage', fontsize=14)
  # plot.xlabel('Time', fontsize=14)
  
  fig = plot.gcf()
  fig.set_size_inches(4, 1)
  plot.savefig('plot_rectangular_TNormal.png', bbox_inches='tight')
  fig.clear()
  # '''
