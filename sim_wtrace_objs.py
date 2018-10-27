import random, simpy, json, queue
import numpy as np
from operator import itemgetter

from utils import is_any_nan
from rvs import *
from log_utils import *

def min_pod_lifetime(pod):
  return pod.cpu_totaldemand/pod.mean_cpu_demandperslot*SCHING_ROUND_LENGTH

def get_pod_usageprofile_m(pod_usageprofile_m_furl):
  with open(pod_usageprofile_m_furl) as f:
    m = json.load(f)
  
  pod_usageprofile_m = {}
  for pod, profile_m in m.items():
    if is_any_nan([
      profile_m['cpu']['mean'], profile_m['memory']['mean'], \
      profile_m['cpu']['reqed'], profile_m['memory']['reqed'] ] ) or \
      (len(profile_m['cpu']['old_mean_l'] ) == 0) or (len(profile_m['memory']['old_max_l'] ) == 0):
      # print('something is wrong!')
      # blog(profile_m=profile_m)
      # blog(mem_req_ed=profile_m['memory']['reqed'] )
      continue
    pod_usageprofile_m[pod] = profile_m
  log(INFO, "", num_pods=len(pod_usageprofile_m) )
  return pod_usageprofile_m

def avg_cpu_mem_totaldemand(pod_usageprofile_m_furl, lifetime_rv):
  pod_usageprofile_m = get_pod_usageprofile_m(pod_usageprofile_m_furl)
  log(INFO, "len(pod_usageprofile_m)= {}".format(len(pod_usageprofile_m) ) )
  lifetime_mean = lifetime_rv.mean()
  
  cpu_totaldemand_l, mem_totaldemand_l = [], []
  for pod, profile_m in pod_usageprofile_m.items():
    # blog(cpu_mean=profile_m['cpu']['mean'], cpu_max=profile_m['cpu']['max'], mem_max=profile_m['memory']['max'], mem_mean=profile_m['memory']['mean'] )
    cpu_totaldemand_l.append(profile_m['cpu']['mean']*lifetime_mean)
    mem_totaldemand_l.append(profile_m['memory']['mean']*lifetime_mean)
  return np.mean(cpu_totaldemand_l), np.mean(mem_totaldemand_l)

def avg_cpu_mem_totaldemand_wrt_requested(pod_usageprofile_m_furl, lifetime_rv, w_prediction=False):
  pod_usageprofile_m = get_pod_usageprofile_m(pod_usageprofile_m_furl)
  # log(INFO, "len(pod_usageprofile_m)= {}".format(len(pod_usageprofile_m) ) )
  lifetime_mean = lifetime_rv.mean()
  
  cpu_totaldemand_l, mem_totaldemand_l = [], []
  for pod, profile_m in pod_usageprofile_m.items():
    if w_prediction:
      cpu_reqed = np.mean(profile_m['cpu']['old_mean_l'] ) + 3*max(profile_m['cpu']['old_std_l'] )
      mem_reqed = max(profile_m['memory']['old_max_l'] )
    else:
      cpu_reqed = profile_m['cpu']['reqed']
      mem_reqed = profile_m['memory']['reqed']
    cpu_totaldemand_l.append(cpu_reqed*lifetime_mean)
    mem_totaldemand_l.append(mem_reqed*lifetime_mean)
  return np.mean(cpu_totaldemand_l), np.mean(mem_totaldemand_l)

# ###########################################  Pod*  ############################################# #
class Pod(object):
  def __init__(self, _id, cpu_demandperslot_rv, mem_demandperslot_rv, cpu_totaldemand, cpu_reqed, mem_reqed):
    self._id = _id
    self.cpu_demandperslot_rv = cpu_demandperslot_rv
    self.mem_demandperslot_rv = mem_demandperslot_rv
    self.cpu_totaldemand = cpu_totaldemand
    self.cpu_reqed = cpu_reqed
    self.mem_reqed = mem_reqed
    
    self.mean_cpu_demandperslot = self.cpu_demandperslot_rv.mean()
    self.mean_mem_demandperslot = self.mem_demandperslot_rv.mean()
    
    self.cpu_cum_supply = 0
    self.cpu_cum_demand = 0
    self.mem_cum_supply = 0
    self.mem_cum_demand = 0
    
    self.bindingt = None
  
  def __repr__(self):
    return "Pod[id= {}]".format(self._id)
  
  def gen_demand(self):
    d = self.cpu_demandperslot_rv.sample()
    remaining = self.cpu_totaldemand - self.cpu_cum_demand
    cpu_demand = d if d < remaining else remaining
    self.cpu_cum_demand += cpu_demand
    
    mem_demand = self.mem_demandperslot_rv.sample()
    # if mem_demand > self.mem_demandperslot_rv.u_l:
    #   log(WARNING, "mem_demand > mem_demandperslot_rv.u_l !!!;", mem_demand=mem_demand, mem_demandperslot_rv_maxval=self.mem_demandperslot_rv.u_l)
    self.mem_cum_demand += mem_demand
    return cpu_demand, mem_demand
  
  def take_supply(self, resource, s):
    if resource == 'cpu':
      s_ = min(self.cpu_cum_demand - self.cpu_cum_supply, s)
      self.cpu_cum_supply += s_
      return s_
    elif resource == 'mem':
      # s_ = min(self.mem_cum_demand - self.mem_cum_supply, s)
      self.mem_cum_supply += s
      return s

class Pod_wdynamicreqed(Pod):
  def __init__(self, _id, cpu_demandperslot_rv, mem_demandperslot_rv, cpu_totaldemand,
               cpu_reqed, mem_reqed, reqed_duration):
    super().__init__(_id, cpu_demandperslot_rv, mem_demandperslot_rv, cpu_totaldemand, cpu_reqed, mem_reqed)
    self.reqed_duration = reqed_duration
    
    self.reqed_countdown = reqed_duration
    self.cpu_demand_max, self.mem_demand_max = 0, 0
    
    cpu_demand_l, mem_demand_l = [], []
  
  def __repr__(self):
    return "Pod_wdynamicreqed[id= {}]".format(self._id)
  
  def gen_demand(self):
    cpu_demand, mem_demand = super().gen_demand()
    
    if cpu_demand > self.cpu_demand_max:
      self.cpu_demand_max = cpu_demand
    if mem_demand > self.mem_demand_max:
      self.mem_demand_max = mem_demand
    
    self.reqed_countdown -= SCHING_ROUND_LENGTH
    if self.reqed_countdown <= 0:
      self.reqed_countdown = self.reqed_duration
      
      self.cpu_reqed = self.cpu_demand_max
      self.cpu_reqed = self.mem_demand_max

class PodGen_fromfile(object):
  def __init__(self, env, ar, pod_usageprofile_m_furl, out, lifetime_rv=None, w_prediction=False, w_dynamicreqed=False, **kwargs):
    self.env = env
    self.ar = ar
    self.out = out
    self.lifetime_rv = lifetime_rv
    self.w_prediction = w_prediction
    self.w_dynamicreqed = w_dynamicreqed
    
    self.pod_usageprofile_m = get_pod_usageprofile_m(pod_usageprofile_m_furl)
    self.pod__profile_m_l = list(self.pod_usageprofile_m.items() )
    
    env.process(self.run() )
    # env.process(self.run_wdynamicreqed() )
    
    self.nsent = 0
  
  def run(self):
    while True:
      yield self.env.timeout(random.expovariate(self.ar) )
      
      pod, profile_m = random.choice(self.pod__profile_m_l)
      self.nsent += 1
      
      if self.w_prediction:
        # log(WARNING, "", cpu_old_mean_l=profile_m['cpu']['old_mean_l'], mem_old_max_l=profile_m['memory']['old_max_l'] )
        cpu_reqed = np.mean(profile_m['cpu']['old_mean_l'] ) # + 3*max(profile_m['cpu']['old_std_l'] )
        mem_reqed = max(profile_m['memory']['old_max_l'] )
        # mem_reqed = np.mean(profile_m['memory']['old_mean_l'] )
      else:
        cpu_reqed = profile_m['cpu']['reqed']
        mem_reqed = profile_m['memory']['reqed']
      # cpu_reqed = profile_m['cpu']['mean'] # profile_m['cpu']['max']
      # mem_reqed = profile_m['memory']['mean'] # profile_m['memory']['max']
      
      lifetime = self.lifetime_rv.sample() if self.lifetime_rv is not None else profile_m['lifetime']
      
      if profile_m['cpu']['stationary'] == 1:
        cpu_demandperslot_rv = TNormal(profile_m['cpu']['mean'], profile_m['cpu']['std'], profile_m['cpu']['max'] )
        # mem_demandperslot_rv = TNormal(profile_m['memory']['mean'], profile_m['memory']['std'], profile_m['memory']['max'] )
      else:
        cpu_demandperslot_rv = gen_rectangular_TNormal(
          mu=profile_m['cpu']['mean'], sigma=profile_m['cpu']['std'], maxval=profile_m['cpu']['max'], duration_mean=lifetime/4)
      if profile_m['memory']['stationary'] == 1:
        mem_demandperslot_rv = TNormal(profile_m['memory']['mean'], profile_m['memory']['std'], profile_m['memory']['max'] )
      else:
        mem_demandperslot_rv = gen_rectangular_TNormal(
          mu=profile_m['memory']['mean'], sigma=profile_m['memory']['std'], maxval=profile_m['memory']['max'], duration_mean=lifetime/4)
      
      if self.w_dynamicreqed:
        p = Pod_wdynamicreqed(
          self.nsent, cpu_demandperslot_rv, mem_demandperslot_rv,
          cpu_demandperslot_rv.mean()*lifetime, cpu_reqed, mem_reqed, reqed_duration=lifetime/3)
      else:
        cpu_total_demand = cpu_demandperslot_rv.mean()*lifetime
        p = Pod(
          self.nsent, cpu_demandperslot_rv, mem_demandperslot_rv,
          cpu_total_demand, cpu_reqed, mem_reqed)
        # print("\n")
        # blog(lifetime=lifetime, cpu_demandperslot_rv_mean = cpu_demandperslot_rv.mean(),
        #     cpu_total_demand=cpu_total_demand, min_lifetime=min_pod_lifetime(p) )
      # print("pushing out pod with lifetime= {}".format(lifetime) )
      self.out.put(p)
  
class PodGen(object):
  def __init__(self, env, ar, cpu_demandmean_rv, mem_demandmean_rv, cpu_totaldemand_rv, out, w_discretereqed=False, **kwargs):
    self.env = env
    self.ar = ar
    self.cpu_demandmean_rv = cpu_demandmean_rv
    self.mem_demandmean_rv = mem_demandmean_rv
    self.cpu_totaldemand_rv = cpu_totaldemand_rv
    self.out = out
    self.w_discretereqed = w_discretereqed
    
    self.nsent = 0
    env.process(self.run() )
  
  def cpu_mem_reqed(self, cpu_demand, mem_demand):
    cpu_leq, mem_leq = None, None
    # cpu_leqdemand_l = [0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 12]
    # mem_leqdemand_l = [0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16]
    cpu_leqdemand_l = [0.5, 1, 2, 4, 8, 12]
    mem_leqdemand_l = [0.5, 1, 2, 4, 8, 12, 16]
    for e in cpu_leqdemand_l:
      if cpu_demand <= e:
        cpu_leq = e
        break
    for e in mem_leqdemand_l:
      if mem_demand <= e:
        mem_leq = e
        break
    if cpu_leq is None or mem_leq is None:
      log(WARNING, "got unclassified values;", cpu_demand=cpu_demand, mem_demand=mem_demand)
    else:
      return cpu_leq, mem_leq
  
  def run(self):
    while True:
      yield self.env.timeout(random.expovariate(self.ar) )
      
      cpu_demandmean = self.cpu_demandmean_rv.sample()
      mem_demandmean = self.mem_demandmean_rv.sample()
      coeff_var = 0.2 # np.random.uniform(0, 1) + 0.01
      cpu_demandstd = coeff_var*cpu_demandmean
      mem_demandstd = coeff_var*mem_demandmean
      cpu_max = cpu_demandmean + 2*cpu_demandstd
      mem_max = mem_demandmean + 2*mem_demandstd
      
      if self.w_discretereqed:
        cpu_reqed, mem_reqed = self.cpu_mem_reqed(cpu_max, mem_max)
      else:
        cpu_reqed, mem_reqed = cpu_max, mem_max
      # blog(cpu_max=cpu_max, mem_max=mem_max, cpu_reqed=cpu_reqed, mem_reqed=mem_reqed)
      
      self.nsent += 1
      self.out.put(Pod(
        _id=self.nsent,
        cpu_demandperslot_rv=TNormal(cpu_demandmean, cpu_demandstd, cpu_max),
        mem_demandperslot_rv=TNormal(mem_demandmean, mem_demandstd, cpu_max),
        cpu_totaldemand=self.cpu_totaldemand_rv.sample(),
        cpu_reqed=cpu_reqed, mem_reqed=mem_reqed) )
  
# #########################################  Worker  ############################################# #
SCHING_ROUND_LENGTH = 1
class Worker(object):
  def __init__(self, env, _id, cpu_cap, mem_cap, out_c=None):
    self.env = env
    self._id = _id
    self.cpu_cap = cpu_cap
    self.mem_cap = mem_cap
    self.out_c = out_c
    
    self.schinground_length = SCHING_ROUND_LENGTH
    self.p_l = []
    self.gotbusy_event = None
    env.process(self.run() )
    
    self.t__cpu_actual_load_m = {}
    self.t__mem_actual_load_m = {}
    
    self.t__cpu_avail_cap_window = queue.Queue(maxsize=100)
    self.t__mem_avail_cap_window = queue.Queue(maxsize=100)
  
  def __repr__(self):
    return "Worker[id= {}]".format(self._id)
  
  def sched_cap(self, res):
    if len(self.p_l) == 0:
      return 0
    if res == 'cpu':
      return sum([p.cpu_reqed for p in self.p_l] )
    elif res == 'mem':
      return sum([p.mem_reqed for p in self.p_l] )
  
  def nonsched_cap(self, res):
    if res == 'cpu':
      cap = self.cpu_cap
    elif res == 'mem':
      cap = self.mem_cap
    return cap - self.sched_cap(res)
  
  def sched_load(self):
    return max(self.sched_cap('cpu')/self.cpu_cap, self.sched_cap('mem')/self.mem_cap)
  
  def avail_cap(self, res, for_last_x_secs=60*5):
    if res == 'cpu':
      t_availcap_window = self.t__cpu_avail_cap_window
      cap = self.cpu_cap
    elif res == 'mem':
      t_availcap_window = self.t__mem_avail_cap_window
      cap = self.mem_cap
    cur_t = self.env.now
    availcap_l = [c for (t, c) in list(t_availcap_window.queue) if cur_t - t <= for_last_x_secs]
    # return min(availcap_l) if len(availcap_l) != 0 else cap
    return np.mean(availcap_l) if len(availcap_l) != 0 else cap
  
  def actual_load(self):
    return max(1 - self.avail_cap('cpu')/self.cpu_cap, 1 - self.avail_cap('mem')/self.mem_cap)
  
  def reg_avail_cap(self, cpu_cap, mem_cap):
    if self.t__cpu_avail_cap_window.full():
      self.t__cpu_avail_cap_window.get()
      self.t__mem_avail_cap_window.get()
    self.t__cpu_avail_cap_window.put((self.env.now, cpu_cap) )
    self.t__mem_avail_cap_window.put((self.env.now, mem_cap) )
  
  def run(self):
    while True:
      yield (self.env.timeout(self.schinground_length) )
      
      if len(self.p_l) == 0:
        slog(DEBUG, self.env, self, "idle; waiting for pod ...", None)
        time_gotidle = self.env.now
        self.gotbusy_event = self.env.event()
        yield (self.gotbusy_event)
        self.gotbusy_event = None
        slog(DEBUG, self.env, self, "got busy!", None)
        
        self.t__cpu_actual_load_m[time_gotidle] = 0
        self.t__cpu_actual_load_m[self.env.now] = 0
        self.t__mem_actual_load_m[time_gotidle] = 0
        self.t__mem_actual_load_m[self.env.now] = 0
        self.reg_avail_cap(self.cpu_cap, self.mem_cap)
        
      for p in self.p_l:
        p.gen_demand()
      
      ### CPU scheduling
      cpu_totalreqed = self.sched_cap('cpu')
      cpu_totalsupplytaken = 0
      for p in self.p_l:
        cpu_totalsupplytaken += p.take_supply(
          'cpu', min(p.cpu_reqed, p.cpu_reqed/cpu_totalreqed*self.cpu_cap) )
      
      p_l_ = self.p_l
      while self.cpu_cap - cpu_totalsupplytaken > 0.01:
        p_l_ = [p for p in p_l_ if p.cpu_cum_demand - p.cpu_cum_supply > 0.01]
        if len(p_l_) == 0:
          break
        
        supply_foreach = (self.cpu_cap - cpu_totalsupplytaken)/len(p_l_)
        for p in p_l_:
          cpu_totalsupplytaken += p.take_supply('cpu', supply_foreach)
      
      # self.t__cpu_sched_load_m[self.env.now] = cpu_totalreqed/self.cpu_cap
      self.t__cpu_actual_load_m[self.env.now] = cpu_totalsupplytaken/self.cpu_cap
      
      ### Memory scheduling
      mem_totalsupplytaken = 0
      for p in self.p_l:
        mem_totalsupplytaken += p.take_supply('mem', p.mem_reqed)
      
      if mem_totalsupplytaken < self.mem_cap:
        # Check if any is asking for more
        podi_l = [i for i, p in enumerate(self.p_l) if p.mem_cum_demand - p.mem_cum_supply > 0.01 ]
        # TODO: (for now) leftover memory cap is given based on first-come first-serve policy
        i = 0
        while i < len(podi_l) and self.mem_cap > mem_totalsupplytaken:
          mem_totalsupplytaken += self.p_l[podi_l[i] ].take_supply('mem', self.mem_cap - mem_totalsupplytaken)
          i += 1
      # Check if any pod is still starving memory
      podi_toevict_l = []
      for i, p in enumerate(self.p_l):
        if p.mem_cum_demand - p.mem_cum_supply > 0.01:
          podi_toevict_l.append(i)
          slog(DEBUG, self.env, self, "evicted", p, mem_cum_demand=p.mem_cum_demand, mem_cum_supply=p.mem_cum_supply)
          self.out_c.put_c({
            'event': 'evicted', 'id': p._id,
            'min_lifetime': min_pod_lifetime(p),
            'runtime': self.env.now - p.bindingt} )
      self.p_l = [p for i, p in enumerate(self.p_l) if i not in podi_toevict_l]
      
      # self.t__mem_actual_load_m[self.env.now] = mem_totalreqed/self.mem_cap
      self.t__mem_actual_load_m[self.env.now] = mem_totalsupplytaken/self.mem_cap
      
      self.reg_avail_cap(self.cpu_cap - cpu_totalsupplytaken, self.mem_cap - mem_totalsupplytaken)
      
      # Check if any of the pods is finished
      p_l_ = []
      for p in self.p_l:
        if p.cpu_cum_supply - p.cpu_totaldemand > -0.01:
          if self.out_c is not None:
            slog(DEBUG, self.env, self, "finished", p)
            self.out_c.put_c(
              {'event': 'finished', 'id': p._id,
               'min_lifetime': min_pod_lifetime(p),
               'runtime': self.env.now - p.bindingt} )
        else:
          p_l_.append(p)
      self.p_l = p_l_
  
  def put(self, p):
    p.bindingt = self.env.now
    _l = len(self.p_l)
    self.p_l.append(p)
    if _l == 0 and self.gotbusy_event is not None:
      self.gotbusy_event.succeed()
    
    slog(DEBUG, self.env, self, "binded, npod= {}".format(len(self.p_l) ), p)

# #########################################  Cluster  ############################################ #
class Cluster(object):
  def __init__(self, env, name, npod, nworker, wcpu_cap, wmem_cap, scher, **kwargs):
    self.env = env
    self.name = name
    self.npod = npod
    self.scher = scher
    
    self.w_l = [Worker(env, i, wcpu_cap, wmem_cap, out_c=self) for i in range(nworker) ]
    
    self.store_c = simpy.Store(env)
    self.wait_for_allpods = env.process(self.run_c() )
    
    self.event_count_m = {}
    self.slowdown_l = []
  
  def __repr__(self):
    return "Cluster[name= {}]".format(self.name)
  
  def put(self, p):
    slog(DEBUG, self.env, self, "received", p)
    a, w = self.scher.map_pod_to_worker(p, self.w_l)
    if a == ACT_BIND:
      w.put(p)
    else:
      self.store_c.put({'id': p._id, 'event': 'dropped', 'cpu_reqed': p.cpu_reqed, 'mem_reqed': p.mem_reqed} )
  
  def run_c(self):
    while True:
      m = yield self.store_c.get()
      e = m['event']
      if e == 'finished':
        self.slowdown_l.append(m['runtime']/m['min_lifetime'] )
      # elif e == 'dropped':
      #   log(WARNING, "dropped;", cpu_reqed=m['cpu_reqed'], mem_reqed=m['mem_reqed'] )
      
      if e not in self.event_count_m:
        self.event_count_m[e] = 0
      self.event_count_m[e] += 1
  
  def put_c(self, m):
    slog(DEBUG, self.env, self, "received", m)
    return self.store_c.put(m)

# ########################################  Scheduler  ########################################### #
ACT_BIND = 1
ACT_DROP = 0
class Scher(object):
  def __init__(self, sching_m):
    self.sching_m = sching_m
    self.sching_wrt = self.sching_m['wrt'] 
    
    self.cpu_reqedcap_l = []
    self.mem_reqedcap_l = []
    
    if self.sching_m['type'] == 'packing':
      self.map_pod_method = lambda p, w_l: self.map_w_packing(p, w_l)
    elif self.sching_m['type'] == 'spreading':
      self.map_pod_method = lambda p, w_l: self.map_w_spreading(p, w_l)
    elif self.sching_m['type'] == 'adaptive_packing':
      self.map_pod_method = lambda p, w_l: self.map_w_adaptive_packing(p, w_l)
    else:
      log(ERROR, "unknown sching_m['type'];", sching_m=self.sching_m)
      return None
  
  def __repr__(self):
    return "Scher[sching_m=\n {}]".format(self.sching_m)
  
  def map_pod_to_worker(self, p, w_l):
    return self.map_pod_method(p, w_l)
  
  def does_w_have_enough(self, p, w):
    if self.sching_wrt == 'sched':
      return (p.cpu_reqed <= w.nonsched_cap('cpu') and p.mem_reqed <= w.nonsched_cap('mem') )
    elif self.sching_wrt == 'actual':
      return (p.cpu_reqed <= w.avail_cap('cpu') and p.mem_reqed <= w.avail_cap('mem') )
  
  def map_w_packing(self, p, w_l):
    for w in w_l:
      if self.does_w_have_enough(p, w):
        return ACT_BIND, w
    return ACT_DROP, None
  
  def map_w_spreading(self, p, w_l):
    w_load_l = []
    for w in w_l:
      if self.does_w_have_enough(p, w):
        if self.sching_wrt == 'sched':
          load = w.sched_load()
        elif self.sching_wrt == 'actual':
          load = w.actual_load()
        w_load_l.append((w, load) )
    if len(w_load_l) == 0:
      return ACT_DROP, None
    
    w, _ = min(w_load_l, key=itemgetter(1) )
    return ACT_BIND, w
  
  def map_w_adaptive_packing(self, p, w_l):
    self.cpu_reqedcap_l.append(p.cpu_reqed)
    self.mem_reqedcap_l.append(p.mem_reqed)
    
    cv_cpu_reqedcap = np.std(self.cpu_reqedcap_l)/np.mean(self.cpu_reqedcap_l)
    cv_mem_reqedcap = np.std(self.mem_reqedcap_l)/np.mean(self.mem_reqedcap_l)
    
    cpu_wavailcap_l = [w.nonsched_cap('cpu') for w in w_l]
    mem_wavailcap_l = [w.nonsched_cap('mem') for w in w_l]
    
    wi__cv_cpu_availcap_l, wi__cv_mem_availcap_l = [], []
    for wi_tobemapped in range(len(cpu_wavailcap_l) ):
      if p.cpu_reqed <= cpu_wavailcap_l[wi_tobemapped] and p.mem_reqed <= mem_wavailcap_l[wi_tobemapped]:
        cpu_wavailcap_l_, mem_wavailcap_l_ = [], []
        for j in range(len(cpu_wavailcap_l) ):
          cpu_wavailcap_l_.append(
            cpu_wavailcap_l[j] + p.cpu_reqed if j == wi_tobemapped else cpu_wavailcap_l[j] )
          mem_wavailcap_l_.append(
            mem_wavailcap_l[j] + p.mem_reqed if j == wi_tobemapped else mem_wavailcap_l[j] )
        cv_cpu_availcap = np.std(cpu_wavailcap_l_)/np.mean(cpu_wavailcap_l_)
        cv_mem_availcap = np.std(mem_wavailcap_l_)/np.mean(mem_wavailcap_l_)
      else:
        cv_cpu_availcap = np.nan
        cv_mem_availcap = np.nan
      wi__cv_cpu_availcap_l.append(cv_cpu_availcap)
      wi__cv_mem_availcap_l.append(cv_mem_availcap)
    
    cv_cpu_diff_l = [abs(cv_cpu_reqedcap - cv) for cv in wi__cv_cpu_availcap_l]
    cv_mem_diff_l = [abs(cv_mem_reqedcap - cv) for cv in wi__cv_mem_availcap_l]
    cv_diff_l = np.array(cv_cpu_diff_l) + np.array(cv_mem_diff_l)
    wi, cv_diff = min(enumerate(cv_diff_l), key=itemgetter(1) )
    if np.isnan(cv_diff):
      return ACT_DROP, None
    return ACT_BIND, w_l[wi]
  