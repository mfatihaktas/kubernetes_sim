from collections import namedtuple
import time, datetime

from data_utils import *
from plot_utils import *
from config import *

class Worker(object):
  def __init__(self, _id, cpu_cap, mem_cap):
    self._id = _id
    self.cpu_cap = cpu_cap
    self.mem_cap = mem_cap
    
    self.pname_profile_m = {}
  
  def avail_cap(self, res, begin_t):
    total_reqed = 0
    for _, profile_m in self.pname_profile_m.items():
      departure_t = None
      if 'completed_t' in profile_m:
        departure_t = profile_m['completed_t']
      elif 'evicted_t' in profile_m:
        departure_t = profile_m['evicted_t']
      if departure_t is not None and \
         begin_t > departure_t:
        pass
      else:
        if res == 'cpu':
          reqed = profile_m['cpu_req']
        elif res == 'memory':
          reqed = profile_m['mem_req']
        total_reqed += reqed
    if res == 'cpu':
      cap = self.cpu_cap
    elif res == 'memory':
      cap = self.mem_cap
    return cap - total_reqed
  
  def bind(self, pname, profile_m):
    self.pname_profile_m[pname] = profile_m
    log(DEBUG, "binded;", pname=pname)
  
  def remove(self, pname):
    self.pname_profile_m.pop(pname, None)
    log(DEBUG, "removed;", pname=pname)

class Cluster(object):
  def __init__(self, nworker, wcpu_cap, wmem_cap, mdatabender, wpredict):
    self.mdatabender = mdatabender
    self.wpredict = wpredict
    
    self.pname_alive_l = []
    self.w_l = [Worker(i, wcpu_cap, wmem_cap) for i in range(nworker) ]
    
    self.pname_profile_m = {}
    self.pname_trie = pytrie.StringTrie()
    
    self.pnameoffthenbackalive_fate_m = {}
    self.datetime_l = []
  
  def __repr__(self):
    return 'Cluster[wpredict= {}]'.format(self.wpredict)
  
  def get_prevmatchingpnames(self, pname):
    longest_prefix_match = lambda prefix: self.pname_trie.values(prefix)
    for i in range(len(pname), -1, -1):
      oldpname_l = longest_prefix_match(pname[:i] )
      if len(oldpname_l) > 0:
        break
    return oldpname_l
  
  def cpu_mem_req_lim(self, pname, _info_m):
    if not self.wpredict:
      cpu_req = _info_m['cpu']['requested']['mean']
      cpu_lim = _info_m['cpu']['limited']['mean']
      # mem_req = _info_m['memory']['limited']['mean']
      mem_req = _info_m['memory']['joe_requested']
      mem_lim = _info_m['memory']['limited']['mean']
      return cpu_req, cpu_lim, mem_req, mem_lim
    
    oldpname_l = self.get_prevmatchingpnames(pname)
    log(INFO, "", pname=pname, oldpname_l=oldpname_l)
    # Filter the old pods in the history w.r.t. requested and limited values
    oldpname_l_ = oldpname_l
    # oldpname_l_ = []
    # for oldpname in oldpname_l:
    #   oldprofile_m = self.pname_profile_m[oldpname]
    #   if 'datetime_info_m' in oldprofile_m:
    #     for _, info_m in oldprofile_m['datetime_info_m'].items():
    #       if abs(_info_m['cpu']['requested']['mean'] - info_m['cpu']['requested']['mean'] ) < 0.1 and \
    #         abs(_info_m['memory']['limited']['mean'] - info_m['memory']['limited']['mean'] ) < 0.1:
    #         oldpname_l_.append(oldpname)
    #         break
    oldprofile_m_l = [self.pname_profile_m[p] for p in oldpname_l_]
    
    print("len(oldprofile_m_l)= {}".format(len(oldprofile_m_l) ) )
    cpu_mean_l, cpu_std_l, cpu_95th_l, cpu_max_l = [], [], [], []
    mem_max_l = []
    def add(info_m):
      cpu_mean_l.append(info_m['cpu']['usage']['mean'] )
      cpu_std_l.append(info_m['cpu']['usage']['std'] )
      cpu_95th_l.append(info_m['cpu']['usage']['95thpercentile'] )
      cpu_max_l.append(info_m['cpu']['usage']['max'] )
      mem_max_l.append(info_m['memory']['current']['max'] )
    
    if len(oldprofile_m_l) > 0:
      for oldprofile_m in oldprofile_m_l:
        if 'datetime_info_m' in oldprofile_m:
          for _, info_m in oldprofile_m['datetime_info_m'].items():
            add(info_m)
        elif 'prev' in oldprofile_m:
        # if 'prev' in oldprofile_m:
          for inner_oldprofile_m in oldprofile_m['prev']:
            if 'datetime_info_m' in inner_oldprofile_m:
              for _, info_m in inner_oldprofile_m['datetime_info_m'].items():
                add(info_m)
    print("len(cpu_max_l)= {}".format(len(cpu_max_l) ) )
    if len(cpu_max_l) > 3:
      cpu_req = np.mean(cpu_mean_l) + 3*np.mean(cpu_std_l) # 1.2*max(cpu_max_l) # max(cpu_95th_l) # np.median(cpu_95th_l) # max(cpu_max_l)
      cpu_lim = np.inf # 2*cpu_req
      mem_req = 1.2*max(mem_max_l)
      mem_lim = np.inf # mem_req
      return cpu_req, cpu_lim, mem_req, mem_lim
    else:
      return np.nan, np.nan, np.nan, np.nan
  
  def add_ashistory(self, pname_profile_m, datetime):
    for pname, profile_m in pname_profile_m.items():
      if is_any_nan([profile_m['cpu']['usage']['mean'], profile_m['memory']['current']['mean'] ] ):
        continue
      
      if pname not in self.pname_profile_m:
        self.pname_trie[pname] = pname
        self.pname_profile_m[pname] = {'fate': {'event': 'history'}, 'datetime_info_m': {} }
      self.pname_profile_m[pname]['datetime_info_m'][datetime] = profile_m
  
  def place_pod(self, pname, info_m, datetime):
    if is_any_nan([info_m['cpu']['usage']['mean'], info_m['memory']['current']['mean'], info_m['cpu']['end_t'] ] ):
      return
    cpu_req, cpu_lim, mem_req, mem_lim = self.cpu_mem_req_lim(pname, info_m)
    # blog(cpu_req=cpu_req, cpu_lim=cpu_lim, mem_req=mem_req, mem_lim=mem_lim)
    if is_any_nan([cpu_req, cpu_lim, mem_req, mem_lim] ):
      cpu_req = info_m['cpu']['requested']['mean']
      cpu_lim = info_m['cpu']['limited']['mean']
      mem_req = info_m['memory']['joe_requested']
      # mem_req = info_m['memory']['limited']['mean']
      mem_lim = info_m['memory']['limited']['mean']
      if is_any_nan([cpu_req, cpu_lim, mem_req, mem_lim] ):
        return
    
    if pname in self.pname_profile_m: # can happen when there is an hour off in pod's metric data is missing
      profile_m = self.pname_profile_m[pname]
      # if 'fate' in profile_m:
      #   fate = profile_m['fate']['event']
      #   if fate != 'history':
      #     log(WARNING, "already in pname_profile_m;", pname=pname)
      #     self.pnameoffthenbackalive_fate_m[pname] = fate
      #     return
      # print("profile_m= {}".format(pprint.pformat(profile_m) ) )
      fate = profile_m['fate']['event']
      if fate == 'evicted':
        return
      elif fate == 'completed':
        self.pnameoffthenbackalive_fate_m[pname] = profile_m['fate']['event']
        if 'prev' not in profile_m:
          profile_m['prev'] = []
        profile_m['prev'].append(
          {k: v for k, v in profile_m.items() if k != 'prev'} )
        self.pname_profile_m[pname] = {'prev': profile_m['prev'] }
    else:
      self.pname_trie[pname] = pname
      self.pname_profile_m[pname] = {}
    profile_m = self.pname_profile_m[pname]
    
    '''
    # Spreading
    wi_availcap_l = []
    for i, w in enumerate(self.w_l):
      cpu_avail_cap = w.avail_cap('cpu', info_m['cpu']['begin_t'] )
      mem_avail_cap = w.avail_cap('memory', info_m['memory']['begin_t'] )
      if cpu_avail_cap > cpu_req and mem_avail_cap > mem_req:
        wi_availcap_l.append((i, min(cpu_avail_cap, mem_avail_cap) ) )
    if len(wi_availcap_l) == 0:
      profile_m.update({'fate': {'event': 'dropped', 't': datetime} } )
      log(INFO, "dropped;", pname=pname)
      return
    
    m = {'cpu_req': cpu_req, 'cpu_lim': cpu_lim, 'mem_req': mem_req, 'mem_lim': mem_lim}
    wi, _ = max(wi_availcap_l, key=itemgetter(1) )
    self.w_l[wi].bind(pname, m)
    '''
    m = {'cpu_req': cpu_req, 'cpu_lim': cpu_lim, 'mem_req': mem_req, 'mem_lim': mem_lim}
    wi = 0
    profile_m.update({'wi': wi, 'datetime_info_m': {datetime: {**m, **info_m} } } )
    self.pname_alive_l.append(pname)
  
  def refine_update_profile(self, pname, info_m, cdatetime):
    # log(DEBUG, "HERE HERE HERE")
    cpu_req, cpu_lim, mem_req, mem_lim = self.cpu_mem_req_lim(pname, info_m)
    if is_any_nan([cpu_req, cpu_lim, mem_req, mem_lim] ):
      print("***HERE***")
      cdatetime_ = cdatetime
      cdatetime_ -= datetime.timedelta(hours=1)
      m = dict(self.pname_profile_m[pname]['datetime_info_m'][cdatetime_] )
      m.update(**info_m)
      # self.pname_profile_m[pname]['datetime_info_m'][cdatetime] = m
    else:
      m = {'cpu_req': cpu_req, 'cpu_lim': cpu_lim, 'mem_req': mem_req, 'mem_lim': mem_lim}
    # self.w_l[profile_m['wi'] ].pname_profile_m[pname].update(m)
    self.pname_profile_m[pname]['datetime_info_m'][cdatetime] = {**m, **info_m}
  
  def run(self, fdatetime, tdatetime, namespace_l):
    log(INFO, "")
    # Build up history
    cdatetime = fdatetime
    # for _ in range(48):
    for _ in range(12):
      pname_info_m = self.mdatabender.get_pod_quickprofile_m(cdatetime, cdatetime, namespace_l)
      self.add_ashistory(pname_info_m, cdatetime)
      cdatetime += datetime.timedelta(hours=1)
    # Run sim
    while cdatetime <= tdatetime:
      self.datetime_l.append(cdatetime)
      pname_info_m = self.mdatabender.get_pod_quickprofile_m(cdatetime, cdatetime, namespace_l)
      pname_tobe_removed_l = []
      # Setup for completion
      for pname in self.pname_alive_l:
        completed_t = None
        if pname not in pname_info_m:
          completed_t = 0
        # else:
        #   info_m = pname_info_m[pname]
        #   completed = (info_m['cpu']['end_t'] - info_m['cpu']['begin_t'] < 60*60 - 120)
        #   if np.isnan(info_m['cpu']['end_t'] ):
        #     completed_t = 0
        #   elif (datetime.datetime.fromtimestamp(info_m['cpu']['end_t'] ) - cdatetime).total_seconds() < 60*60*8/10:
        #     self.pname_profile_m[pname]['datetime_info_m'][cdatetime] = info_m # in case needed
        #     completed_t = info_m['cpu']['end_t']
        if completed_t is not None:
          # datetime = cdatetime - datetime.timedelta(hours=1)
          profile_m = self.pname_profile_m[pname]
          # print("pname= {}, profile_m= {}".format(pname, profile_m) )
          profile_m['fate'] = {'event': 'completed', 't': cdatetime}
          # print("pname= {}, profile_m= {}".format(pname, profile_m) )
          # self.w_l[profile_m['wi'] ].pname_profile_m[pname]['completed_t'] = completed_t
          pname_tobe_removed_l.append(pname)
      # Refine requested as if refining is done right before the current hour starts
      for pname in self.pname_alive_l:
        if pname not in pname_info_m: # Pod is completed
          continue
        self.refine_update_profile(pname, pname_info_m[pname], cdatetime)
      '''
      # Setup for eviction
      for pname in self.pname_alive_l:
        if pname not in pname_info_m: # Pod is completed
          continue
        profile_m = self.pname_profile_m[pname]
        info_m = profile_m['datetime_info_m'][cdatetime]
        reason = None
        # if info_m['cpu']['usage']['max'] > info_m['cpu_lim']:
        if info_m['cpu']['usage']['max'] > info_m['cpu_lim']:
          # reason = 'cpu_max > cpu_lim'
          reason = 'cpu_max > cpu_lim'
          # reason = '{}; cpu_95th= {} > cpu_lim= {}'.format(pname, info_m['cpu']['usage']['95thpercentile'], info_m['cpu_lim'] )
          details = '{}; cpu_max= {} > cpu_lim= {}'.format(pname, info_m['cpu']['usage']['max'], info_m['cpu_lim'] )
          evict_t = info_m['cpu']['max_t']
        elif info_m['memory']['current']['max'] > info_m['mem_lim']:
          reason = 'mem_max > mem_lim'
          details = '{}; mem_max= {} > mem_lim= {}'.format(pname, info_m['memory']['current']['max'], info_m['mem_lim'] )
          evict_t = info_m['memory']['max_t']
        if reason is not None:
          profile_m['fate'] = {'event': 'evicted', 't': cdatetime, 'reason': reason, 'details': details}
          self.w_l[profile_m['wi'] ].pname_profile_m[pname]['evicted_t'] = evict_t
          if pname not in pname_tobe_removed_l: # Might have already been added due to completion
            pname_tobe_removed_l.append(pname)
      '''
      # Place newly arriving pods
      for pname, info_m in pname_info_m.items():
        if pname not in self.pname_alive_l:
          self.place_pod(pname, info_m, cdatetime)
      # Merge with the new profile
      # self.add_ashistory(pname_info_m, cdatetime)
      # Execute completions and evictions
      for pname in pname_tobe_removed_l:
        # self.w_l[self.pname_profile_m[pname]['wi'] ].remove(pname)
        self.pname_alive_l.remove(pname)
      
      cdatetime += datetime.timedelta(hours=1)
  
  def inspect(self):
    fate_count_m = {'history': 0, 'dropped': 0, 'completed': 0, 'evicted': 0}
    evicted_reason = {'cpu_max > cpu_lim': 0, 'mem_max > mem_lim': 0}
    evicted_details_l = []
    
    for pname, profile_m in self.pname_profile_m.items():
      s = '>> {}\n'.format(pname)
      if 'fate' in profile_m:
        fate = profile_m['fate']['event']
        s += 'fate= {}\n'.format(fate)
        fate_count_m[fate] += 1
        if fate == 'dropped' or fate == 'history':
          # print(s + '\n')
          continue
        elif fate == 'evicted':
          evicted_reason[profile_m['fate']['reason'] ] += 1
          evicted_details_l.append(profile_m['fate']['details'] )
      # s += 'wi= {}\n'.format(profile_m['wi'] )
      # for datetime, info_m in profile_m['datetime_info_m'].items():
      #   s += 'datetime= {}\n'.format(datetime)
      #   s += 'cpu_avg_util= {}\n'.format(info_m['cpu']['usage']['mean']/info_m['cpu_req'] )
      #   s += 'mem_avg_util= {}\n'.format(info_m['memory']['current']['mean']/info_m['mem_req'] )
      # print(s + '\n')
    
    log(INFO, "", npods=len(self.pname_profile_m), fate_count_m=fate_count_m, \
        evicted_reason=evicted_reason, evicted_details_l=evicted_details_l)
    # log(INFO, "", npods=len(self.pname_profile_m), fate_count_m=fate_count_m, evicted_reason_l=evicted_reason_l)
    # blog(pnameoffthenbackalive_fate_m=cl.pnameoffthenbackalive_fate_m)
  
  def plot_cdf_utilization(self):
    cpu_mean_util_l, cpu_95th_util_l, cpu_max_util_l = [], [], []
    mem_mean_util_l, mem_95th_util_l, mem_max_util_l = [], [], []
    npods = 0
    for pname, profile_m in self.pname_profile_m.items():
      # try:
      #   fate = profile_m['fate']
      # except:
      #   fate = 'ongoing'
      if 'datetime_info_m' not in profile_m:
        continue
      if 'fate' in profile_m and profile_m['fate']['event'] == 'evicted':
        continue
      
      cpu_mean_l, cpu_95th_l, cpu_max_l = [], [], []
      mem_mean_l, mem_95th_l, mem_max_l = [], [], []
      f_l = [lambda e: not np.isnan(e) ]
      for datetime, info_m in profile_m['datetime_info_m'].items():
        if 'cpu_req' in info_m:
          cpu_req = info_m['cpu_req']
          cpu_mean = info_m['cpu']['usage']['mean']/cpu_req
          cpu_95th = info_m['cpu']['usage']['95thpercentile']/cpu_req
          cpu_max = info_m['cpu']['usage']['95thpercentile']/cpu_req
          append_if(cpu_mean_l, cpu_mean, f_l)
          append_if(cpu_95th_l, cpu_95th, f_l)
          append_if(cpu_max_l, cpu_max, f_l)
          # if info_m['cpu']['usage']['95thpercentile'] > info_m['cpu_req']:
          #   blog(pod=pname, cpu_usage_95th=info_m['cpu']['usage']['95thpercentile'], cpu_req=info_m['cpu_req'] )
          
          mem_req = info_m['mem_req']
          mem_mean = info_m['memory']['current']['mean']/mem_req
          mem_95th = info_m['memory']['current']['95thpercentile']/mem_req
          mem_max = info_m['memory']['current']['max']/mem_req
          append_if(mem_mean_l, mem_mean, f_l)
          append_if(mem_95th_l, mem_95th, f_l)
          append_if(mem_max_l, mem_max, f_l)
      if len(cpu_mean_l) == 0:
        continue
      npods += 1
      
      cpu_mean_util = np.mean(cpu_mean_l)
      cpu_95th_util = np.median(cpu_95th_l)
      cpu_max_util = max(cpu_max_l)
      mem_mean_util = np.mean(mem_mean_l)
      mem_95th_util = np.median(mem_95th_l)
      mem_max_util = max(mem_max_l)
      # if cpu_95th_util > 1:
      #   print("pod= {}, cpu_95th_l= {}".format(pname, cpu_95th_l) )
      # if mem_max_util > 1:
      #   print("pod= {}, mem_max_l= {}".format(pname, mem_max_l) )
      
      cpu_mean_util_l.append(cpu_mean_util)
      cpu_95th_util_l.append(cpu_95th_util)
      cpu_max_util_l.append(cpu_max_util)
      mem_mean_util_l.append(mem_mean_util)
      mem_95th_util_l.append(mem_95th_util)
      mem_max_util_l.append(mem_max_util)
    log(INFO, "npods= {}".format(npods) )
    fig, axs = plot.subplots(1, 2)
    figsize = (2*5, 4)
    ax = axs[0]
    plot.sca(ax)
    add_cdf(cpu_mean_util_l, ax, 'Average', NICE_GREEN)
    add_cdf(cpu_95th_util_l, ax, '95th percentile', NICE_BLUE)
    add_cdf(cpu_max_util_l, ax, 'Maximum', NICE_RED, drawline_x_l=[0.1, 1] )
    plot.xscale('log')
    plot.legend()
    plot.xlabel('Utilization of requested CPU', fontsize=16)
    plot.ylabel('CDF', fontsize=16)
    prettify(ax)
    
    ax = axs[1]
    plot.sca(ax)
    add_cdf(mem_mean_util_l, ax, 'Average', NICE_GREEN)
    add_cdf(mem_95th_util_l, ax, '95th percentile', NICE_BLUE)
    add_cdf(mem_max_util_l, ax, 'Maximum', NICE_RED, drawline_x_l=[0.1, 1] )
    plot.xscale('log')
    plot.legend()
    plot.xlabel('Utilization of requested Memory', fontsize=16)
    plot.ylabel('CDF', fontsize=16)
    prettify(ax)
    
    # st = plot.suptitle('Number of pods'.format(fdatetime.strftime('%y/%m/%d %H:%M'), tdatetime.strftime('%y/%m/%d %H:%M') ) )
    fig.set_size_inches(figsize[0], figsize[1] )
    plot.subplots_adjust(hspace=0.45, wspace=0.45)
    fig.patch.set_alpha(0.5)
    plot.savefig('plot_sim_utilization_wpredict{}.png'.format(self.wpredict), bbox_inches='tight') # bbox_extra_artists=(st,)
    fig.clear()
    log(WARNING, "done.")
  
def plot_aggedutilization_overtime():
  # cl_wdefaultreqed = sim_cl(wpredict=False)
  cl_wpredict = sim_cl(wpredict=True)
  
  def get_info_m(pname, cdatetime, cl):
    profile_m = cl.pname_profile_m[pname]
    if 'datetime_info_m' in profile_m:
      for datetime, info_m in profile_m['datetime_info_m'].items():
        if datetime == cdatetime:
          return info_m
    if 'prev' in profile_m:
      profile_m = profile_m['prev']
      if 'datetime_info_m' in profile_m:
        for datetime, info_m in profile_m['datetime_info_m'].items():
          if datetime == cdatetime:
            return info_m
    return None
  
  datetime_info_m = {}
  for cdatetime in cl_wpredict.datetime_l:
    npods = 0
    cpu_cumdefaultreqed, cpu_cumreqed = 0, 0
    cpu_mean_cumusage, cpu_95th_cumusage, cpu_max_cumusage = 0, 0, 0
    mem_cumdefaultreqed, mem_cumreqed = 0, 0
    mem_mean_cumusage, mem_95th_cumusage, mem_max_cumusage = 0, 0, 0
    for pname in cl_wpredict.pname_profile_m:
      info_wpredict_m = get_info_m(pname, cdatetime, cl_wpredict)
      # info_wdefaultreqed_m = get_info_m(pname, cdatetime, cl_wdefaultreqed)
      try:
        if is_any_nan([info_wpredict_m['cpu']['usage']['mean'], info_wpredict_m['memory']['current']['mean'] ] ):
          continue
        
        cpu_defaultreqed = info_wpredict_m['cpu']['requested']['mean'] # info_wdefaultreqed_m['cpu_req']
        mem_defaultreqed = info_wpredict_m['memory']['joe_requested'] # info_wdefaultreqed_m['mem_req']
        if is_any_nan([cpu_defaultreqed, mem_defaultreqed] ):
          continue
        cpu_mean_cumusage += info_wpredict_m['cpu']['usage']['mean']
        cpu_95th_cumusage += info_wpredict_m['cpu']['usage']['95thpercentile']
        cpu_max_cumusage += info_wpredict_m['cpu']['usage']['max']
        cpu_cumdefaultreqed += cpu_defaultreqed
        cpu_cumreqed += info_wpredict_m['cpu_req']
        
        mem_mean_cumusage += info_wpredict_m['memory']['current']['mean']
        mem_95th_cumusage += info_wpredict_m['memory']['current']['95thpercentile']
        mem_max_cumusage += info_wpredict_m['memory']['current']['max']
        mem_cumdefaultreqed += mem_defaultreqed
        mem_cumreqed += info_wpredict_m['mem_req']
        
        npods += 1
      except:
        pass
    # log(INFO, "", cdatetime=cdatetime, npods=npods)
    datetime_info_m[cdatetime] = {
      'cpu': {
        'cumdefaultreqed': cpu_cumdefaultreqed,
        'cumreqed': cpu_cumreqed,
        'mean_cumusage': cpu_mean_cumusage,
        '95th_cumusage': cpu_95th_cumusage,
        'max_cumusage': cpu_max_cumusage},
      'memory': {
        'cumdefaultreqed': mem_cumdefaultreqed,
        'cumreqed': mem_cumreqed,
        'mean_cumusage': mem_mean_cumusage,
        '95th_cumusage': mem_95th_cumusage,
        'max_cumusage': mem_max_cumusage} }
  
  def plot_all(res):
    def try_(a, b):
      try:
        return a/b
      except:
        return np.nan
    # cumdefaultreqed_l, cumreqed_l = [], []
    # mean_cumusage_l, _95th_cumusage_l, max_cumusage_l = [], [], []
    mean_cumutil_wrt_defaultreqed_l, _95th_cumutil_wrt_defaultreqed_l, max_cumutil_wrt_defaultreqed_l = [], [], []
    mean_cumutil_wrt_predicted_l, _95th_cumutil_wrt_predicted_l, max_cumutil_wrt_predicted_l = [], [], []
    for cdatetime in cl_wpredict.datetime_l:
      info_m = datetime_info_m[cdatetime][res]
      # cumdefaultreqed_l.append(info_m['cumdefaultreqed'] )
      # cumreqed_l.append(info_m['cumreqed'] )
      # mean_cumusage_l.append(info_m['mean_cumusage'] )
      # _95th_cumusage_l.append(info_m['95th_cumusage'] )
      # max_cumusage_l.append(info_m['max_cumusage'] )
      
      mean_cumutil_wrt_defaultreqed = try_(info_m['mean_cumusage'], info_m['cumdefaultreqed'] )
      _95th_cumutil_wrt_defaultreqed = try_(info_m['95th_cumusage'], info_m['cumdefaultreqed'] )
      max_cumutil_wrt_defaultreqed = try_(info_m['max_cumusage'], info_m['cumdefaultreqed'] )
      
      mean_cumutil_wrt_predicted = try_(info_m['mean_cumusage'], info_m['cumreqed'] )
      _95th_cumutil_wrt_predicted = try_(info_m['95th_cumusage'], info_m['cumreqed'] )
      max_cumutil_wrt_predicted = try_(info_m['max_cumusage'], info_m['cumreqed'] )
      
      mean_cumutil_wrt_defaultreqed_l.append(mean_cumutil_wrt_defaultreqed)
      _95th_cumutil_wrt_defaultreqed_l.append(_95th_cumutil_wrt_defaultreqed)
      max_cumutil_wrt_defaultreqed_l.append(max_cumutil_wrt_defaultreqed)
      
      mean_cumutil_wrt_predicted_l.append(mean_cumutil_wrt_predicted)
      _95th_cumutil_wrt_predicted_l.append(_95th_cumutil_wrt_predicted)
      max_cumutil_wrt_predicted_l.append(max_cumutil_wrt_predicted)
    lw, mew, ms = 2, 2, 2
    # plot.plot(cl_wpredict.datetime_l, cumdefaultreqed_l, label='Requested w/ default values', color='gray', marker='_', linestyle='-', lw=lw, mew=mew, ms=ms)
    # plot.plot(cl_wpredict.datetime_l, cumreqed_l, label='Requested w/ prediction', color='black', marker='_', linestyle='--', lw=lw, mew=mew, ms=ms)
    # plot.plot(cl_wpredict.datetime_l, mean_cumusage_l, label='Average', color=NICE_GREEN, marker='_', linestyle=':', lw=lw, mew=mew, ms=ms)
    # plot.plot(cl_wpredict.datetime_l, _95th_cumusage_l, label='95th percentile', color=NICE_BLUE, marker='_', linestyle=':', lw=lw, mew=mew, ms=ms)
    # plot.plot(cl_wpredict.datetime_l, max_cumusage_l, label='Maximum', color=NICE_RED, marker='_', linestyle=':', lw=lw, mew=mew, ms=ms)
    plot.plot(cl_wpredict.datetime_l, mean_cumutil_wrt_defaultreqed_l, label='Average', color='gold', marker='x', linestyle=':', lw=lw, mew=mew, ms=ms)
    plot.plot(cl_wpredict.datetime_l, _95th_cumutil_wrt_defaultreqed_l, label='95th percentile', color='gray', marker='+', linestyle=':', lw=lw, mew=mew, ms=ms)
    plot.plot(cl_wpredict.datetime_l, max_cumutil_wrt_defaultreqed_l, label='Maximum', color='mediumpurple', marker='o', linestyle=':', lw=lw, mew=mew, ms=ms)
    
    plot.plot(cl_wpredict.datetime_l, mean_cumutil_wrt_predicted_l, label='Average w/ prediction', color=NICE_GREEN, marker='x', linestyle=':', lw=lw, mew=mew, ms=ms)
    plot.plot(cl_wpredict.datetime_l, _95th_cumutil_wrt_predicted_l, label='95th w/ prediction', color=NICE_BLUE, marker='+', linestyle=':', lw=lw, mew=mew, ms=ms)
    plot.plot(cl_wpredict.datetime_l, max_cumutil_wrt_predicted_l, label='Maximum w/ prediction', color=NICE_RED, marker='o', linestyle=':', lw=lw, mew=mew, ms=ms)
  fontsize = 16
  fig, axs = plot.subplots(2, 1, sharex='col')
  ax = axs[0]
  plot.sca(ax)
  plot_all('cpu')
  # plot.yscale('log')
  plot.legend(loc='best', framealpha=0.5) # 'upper left', 'upper center'
  ax.axes.get_xaxis().set_ticks([])
  # plot.xlabel(fontsize=fontsize)
  plot.ylabel('Utilization of total\nrequested CPU', fontsize=fontsize)
  prettify(ax)
  
  ax = axs[1]
  plot.sca(ax)
  plot_all('memory')
  # plot.yscale('log')
  plot.legend(loc='best', framealpha=0.5)
  ax.axes.get_xaxis().set_ticks([])
  # plot.xticks(rotation=80)
  # plot.xlabel(fontsize=fontsize)
  plot.xlabel('Time', fontsize=fontsize)
  plot.ylabel('Utilization of total\nrequested Memory', fontsize=fontsize)
  prettify(ax)
  
  # fig.set_size_inches(2*(len(cl_wpredict.datetime_l)*0.5), 4)
  fig.set_size_inches(8, 2*4)
  fig.patch.set_alpha(0.5)
  plot.subplots_adjust(wspace=0.2)
  plot.savefig('plot_aggedutilization_overtime.png', bbox_inches='tight')
  fig.clear()
  log(WARNING, "done.")
  
def sim_cl(wpredict):
  pcos = ParseCOS(
    coscredentialsjson_dir=None, bucket=None,
    cluster='prdwat-dal10-cruiser2',
    joe_reqedmetrics_savedir = '/Users/mehmet.aktas/Desktop/COS_scripts/perfworks',
    prefix_dataatdisk='/Users/mehmet.aktas/Desktop')
  mdatabender = MetricDataBender(pcos)
  
  cl = Cluster(nworker=1000, wcpu_cap=100*100, wmem_cap=400*100, mdatabender=mdatabender, wpredict=wpredict)
  cl.run(
    fdatetime = datetime.datetime(2018, 7, 5, 0),
    tdatetime = datetime.datetime(2018, 7, 10, 1),
    namespace_l = cognitive_namespace_l)
  # cl.run(
  #   fdatetime = datetime.datetime(2018, 7, 5, 0),
  #   tdatetime = datetime.datetime(2018, 7, 9, 23),
  #   namespace_l = cognitive_namespace_l)
  # cl.run(
  #   fdatetime = datetime.datetime(2018, 7, 7, 0),
  #   tdatetime = datetime.datetime(2018, 7, 8, 23),
  #   namespace_l = cognitive_namespace_l)
  # cl.run(
  #   fdatetime = datetime.datetime(2018, 7, 5, 0),
  #   tdatetime = datetime.datetime(2018, 7, 12, 23),
  #   namespace_l = cognitive_namespace_l)
  cl.inspect()
  cl.plot_cdf_utilization()
  print("cl= {}".format(cl) )
  return cl

if __name__ == '__main__':
  # sim_cl(wpredict=True)
  # sim_cl(wpredict=False)
  
  plot_aggedutilization_overtime()
  