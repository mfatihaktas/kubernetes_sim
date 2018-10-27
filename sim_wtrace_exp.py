import textwrap, datetime

from utils import *
from plot_utils import *
from sim_wtrace_objs import *

def arrival_rate_upperbound(sinfo_m, w_prediction=False):
  if 'pod_usageprofile_m_furl' in sinfo_m:
    cpu_avg_totaldemand, mem_avg_totaldemand = avg_cpu_mem_totaldemand_wrt_requested(
      sinfo_m['pod_usageprofile_m_furl'], sinfo_m['lifetime_rv'], w_prediction)
  else:
    cpu_avg_totaldemand = sinfo_m['cpu_totaldemand_rv'].mean()
    mem_avg_totaldemand = cpu_avg_totaldemand
  ar_ub_wrt_cpu = sinfo_m['nworker']*sinfo_m['wcpu_cap']/cpu_avg_totaldemand
  ar_ub_wrt_mem = sinfo_m['nworker']*sinfo_m['wmem_cap']/mem_avg_totaldemand
  if ar_ub_wrt_mem < ar_ub_wrt_cpu:
    log(WARNING, 'Sys is Memory bounded')
  else:
    log(WARNING, 'Sys is CPU bounded')
  return min(ar_ub_wrt_cpu, ar_ub_wrt_mem)

def offered_load(sinfo_m, w_prediction=False):
  if 'pod_usageprofile_m_furl' in sinfo_m:
    cpu_avg_totaldemand, mem_avg_totaldemand = avg_cpu_mem_totaldemand_wrt_requested(
      sinfo_m['pod_usageprofile_m_furl'], sinfo_m['lifetime_rv'], w_prediction)
  else:
    cpu_avg_totaldemand = sinfo_m['cpu_totaldemand_rv'].mean()
    mem_avg_totaldemand = cpu_avg_totaldemand
  cpu_load =  sinfo_m['ar']*cpu_avg_totaldemand/sinfo_m['nworker']/sinfo_m['wcpu_cap']
  mem_load =  sinfo_m['ar']*mem_avg_totaldemand/sinfo_m['nworker']/sinfo_m['wmem_cap']
  return round(max(cpu_load, mem_load), 2)

def sim(sinfo_m, sching_m, add_toplotname=''):
  env = simpy.Environment()
  cl = Cluster(env, 'exp', scher=Scher(sching_m), **sinfo_m)
  if 'pod_usageprofile_m_furl' in sinfo_m:
    pg = PodGen_fromfile(env, out=cl, **sinfo_m)
  else:
    pg = PodGen(env, out=cl, **sinfo_m)
  
  env.run(until=sinfo_m['npod']/sinfo_m['ar'] )
  def safe_f(l, f):
    if len(l) == 0:
      return 0
    return f(l)
  
  fig, axs = plot.subplots(len(cl.w_l), 2, sharex='col')
  max_wcpu_util_l, avg_wcpu_util_l = [], []
  max_wmem_util_l, avg_wmem_util_l = [], []
  for i, w in enumerate(cl.w_l):
    color = next(dark_color)
    marker = next(marker_cycle)
    plot.sca(axs[i, 0] )
    t_l, cpu_actual_load_l = map_to_key__val_l(w.t__cpu_actual_load_m)
    plot.plot(t_l, cpu_actual_load_l, label='w.id= {}'.format(w._id), color=color, marker=marker, linestyle=':', mew=2)
    plot.ylabel('CPU actual load')
    plot.legend()
    plot.xticks(rotation=70)
    plot.xlabel('Time (sec)')
    
    plot.sca(axs[i, 1] )
    t_l, mem_actual_load_l = map_to_key__val_l(w.t__mem_actual_load_m)
    plot.plot(t_l, mem_actual_load_l, label='w.id= {}'.format(w._id), color=color, marker=marker, linestyle=':', mew=2)
    plot.ylabel('Memory actual load')
    plot.legend()
    plot.xticks(rotation=70)
    plot.xlabel('Time (sec)')
    
    avg_cpu_load = safe_f(cpu_actual_load_l, np.mean)
    avg_mem_load = safe_f(mem_actual_load_l, np.mean)
    blog(wid=w._id, avg_cpu_load=avg_cpu_load, avg_mem_load=avg_mem_load)
    
    max_wcpu_util_l.append(safe_f(cpu_actual_load_l, max) )
    avg_wcpu_util_l.append(avg_cpu_load)
    max_wmem_util_l.append(safe_f(mem_actual_load_l, max) )
    avg_wmem_util_l.append(avg_mem_load)
  
  plot.gcf().set_size_inches(2*8, len(cl.w_l)*4)
  # plot.subplots_adjust(hspace=0.25)
  savename_suffix = '{}_wrt{}_{}_ar{}'.format(sching_m['type'], sching_m['wrt'], add_toplotname, sinfo_m['ar'] )
  plot.subplots_adjust(hspace=0.25, wspace=0.25)
  plot.savefig('plot_wloadovertime_{}.png'.format(savename_suffix), bbox_inches='tight')
  
  blog(pg_nsent=pg.nsent, cl_event_count_m=cl.event_count_m)
  return {
    'drop_rate': cl.event_count_m['dropped']/pg.nsent if 'dropped' in cl.event_count_m else 0,
    'evict_rate': cl.event_count_m['evicted']/pg.nsent if 'evicted' in cl.event_count_m else 0,
    'avg_slowdown': np.mean(cl.slowdown_l),
    'max_cpu_util': np.mean(max_wcpu_util_l),
    'avg_cpu_util': np.mean(avg_wcpu_util_l),
    'max_mem_util': np.mean(max_wmem_util_l),
    'avg_mem_util': np.mean(avg_wmem_util_l) }

def plot_wrt_ar():
  sinfo_m = {
    'ar': None, 'npod': 40000, 'nworker': 10, 'wcpu_cap': 16, 'wmem_cap': 32,
    'cpu_demandmean_rv': TPareto(0.1, 8, 1.1), 'mem_demandmean_rv': TPareto(0.1, 12, 1.1),
    'cpu_totaldemand_rv': TPareto(1, 10000, 1) }
  blog(sinfo_m=sinfo_m)
  
  ar_ub = arrival_rate_upperbound(sinfo_m)
  print("ar_ub= {}".format(ar_ub) )
  
  ar_offeredload_m = {}
  fig, axs = plot.subplots(1, 3)
  def plot_(sching_m):
    print("sching_m= {}".format(sching_m) )
    ar_l = []
    drop_rate_l, evict_rate_l, avg_slowdown_l = [], [], []
    # for ar in np.linspace(ar_ub/2, ar_ub, 2):
    for ar in [2*ar_ub/3]:
      ar = round(ar, 2)
      sinfo_m['ar'] = ar
      print("\nar= {}".format(ar) )
      ar_l.append(ar)
      
      ol = offered_load(sinfo_m)
      print("offered_load= {}".format(ol) )
      ar_offeredload_m[ar] = ol
      
      sim_m = sim(sinfo_m, sching_m, add_toplotname='wdiscretereqed' if 'w_discretereqed' in sinfo_m else '')
      print("drop_rate= {}".format(sim_m['drop_rate'] ) )
      drop_rate_l.append(sim_m['drop_rate'] )
      print("evict_rate= {}".format(sim_m['evict_rate'] ) )
      evict_rate_l.append(sim_m['evict_rate'] )
      print("avg_slowdown= {}".format(sim_m['avg_slowdown'] ) )
      avg_slowdown_l.append(sim_m['avg_slowdown'] )
    color, marker = next(dark_color), next(marker_cycle)
    label = '{} wrt {}'.format(sching_m['type'], sching_m['wrt'] )
    if 'w_discretereqed' in sinfo_m:
      label += ' w_discretereqed'
    plot.sca(axs[0] )
    plot.plot(ar_l, drop_rate_l, color=color, label=label, marker=marker, linestyle=':', mew=2)
    plot.sca(axs[1] )
    plot.plot(ar_l, evict_rate_l, color=color, label=label, marker=marker, linestyle=':', mew=2)
    # plot.sca(axs[2] )
    # plot.plot(ar_l, avg_slowdown_l, color=color, label=label, marker=marker, linestyle=':', mew=2)
  
  # plot_({'type': 'wrt_coeffvar'} )
  # plot_({'type': 'to_firstavail'} )
  
  # plot_({'type': 'spreading', 'wrt': 'actual'} )
  # plot_({'type': 'spreading', 'wrt': 'sched'} )
  print("*** w_discretereqed")
  sinfo_m['w_discretereqed'] = True
  plot_({'type': 'spreading', 'wrt': 'actual'} )
  
  ax = axs[0]
  plot.sca(ax)
  s = 'sinfo_m= {}'.format(sinfo_m) + \
      '\nar_offeredload_m= {}'.format(ar_offeredload_m)
  an = plot.annotate(
    textwrap.fill(s, 150), xy=(-0.1, -0.2), xycoords='axes fraction', xytext=(-0.1, -0.2), textcoords='axes fraction', va='top')
  plot.legend()
  plot.xlabel('Arrival rate', fontsize=14)
  plot.ylabel('Pod drop rate', fontsize=14)
  ax = axs[1]
  plot.sca(ax)
  plot.legend()
  plot.xlabel('Arrival rate', fontsize=14)
  plot.ylabel('Pod eviction rate', fontsize=14)
  ax = axs[2]
  plot.sca(ax)
  plot.legend()
  plot.xlabel('Arrival rate', fontsize=14)
  plot.ylabel('Avg slowdown', fontsize=14)
  
  plot.gcf().set_size_inches(len(axs)*5, 4)
  plot.subplots_adjust(hspace=0.5)
  plot.savefig("plot_wrt_ar.png", bbox_extra_artists=(an,), bbox_inches='tight')
  log(WARNING, "done.")

def plot_sching_wrt_requested_vs_predicted():
  # history_fdatetime = datetime.datetime(2018, 7, 7, 0)
  # history_tdatetime = datetime.datetime(2018, 7, 7, 23)
  # fdatetime = datetime.datetime(2018, 7, 8, 0)
  # tdatetime = datetime.datetime(2018, 7, 8, 23)
  # history_fdatetime = datetime.datetime(2018, 6, 21, 0)
  # history_tdatetime = datetime.datetime(2018, 6, 21, 0)
  # fdatetime = datetime.datetime(2018, 7, 6, 0)
  # tdatetime = datetime.datetime(2018, 7, 6, 1)
  # history_fdatetime = datetime.datetime(2018, 6, 21, 0)
  # history_tdatetime = datetime.datetime(2018, 6, 23, 23)
  # fdatetime = datetime.datetime(2018, 7, 1, 0)
  # tdatetime = datetime.datetime(2018, 7, 1, 23)
  history_fdatetime = datetime.datetime(2018, 6, 21, 0)
  history_tdatetime = datetime.datetime(2018, 6, 21, 12)
  fdatetime = datetime.datetime(2018, 7, 6, 0)
  tdatetime = datetime.datetime(2018, 7, 6, 12)
  pod_usageprofile_m_furl = 'pod_usageprofileforsim_m_' + \
    '_historyf'+history_fdatetime.strftime('%y%m%d-%H:%M') + \
    '_historyt'+history_tdatetime.strftime('%y%m%d-%H:%M') + '_' + \
    '_f'+fdatetime.strftime('%y%m%d-%H:%M') + \
    '_t'+tdatetime.strftime('%y%m%d-%H:%M') + '.json'
  sinfo_m = {
    'ar': None, 'npod': 40000, 'nworker': 10, 'wcpu_cap': 8, 'wmem_cap': 400,
    'pod_usageprofile_m_furl': pod_usageprofile_m_furl,
    'lifetime_rv': TPareto(10, 10000, 1.1) }
  blog(sinfo_m=sinfo_m)
  
  ar_ub = 12*arrival_rate_upperbound(sinfo_m, w_prediction=True)
  print("ar_ub= {}".format(ar_ub) )
  
  ar_offeredload_m = {}
  fig, axs = plot.subplots(1, 4)
  def plot_(sching_m, w_prediction, label=None):
    sinfo_m['w_prediction'] = w_prediction
    print("sching_m= {}".format(sching_m) )
    
    ar_l = []
    drop_rate_l, evict_rate_l, avg_slowdown_l = [], [], []
    max_cpu_util_l, avg_cpu_util_l = [], []
    max_mem_util_l, avg_mem_util_l = [], []
    for ar in np.linspace(ar_ub/5, ar_ub, 7):
    # for ar in [ar_ub]:
      ar = round(ar, 2)
      sinfo_m['ar'] = ar
      print("\nar= {}".format(ar) )
      ar_l.append(ar)
      
      ol = offered_load(sinfo_m, w_prediction)
      print("offered_load= {}".format(ol) )
      ar_offeredload_m[ar] = ol
      
      sim_m = sim(sinfo_m, sching_m,
        add_toplotname='reqedfromprediction' if w_prediction else 'reqedNOTfromprediction')
      print("drop_rate= {}".format(sim_m['drop_rate'] ) )
      drop_rate_l.append(sim_m['drop_rate'] )
      print("evict_rate= {}".format(sim_m['evict_rate'] ) )
      evict_rate_l.append(sim_m['evict_rate'] )
      print("avg_slowdown= {}".format(sim_m['avg_slowdown'] ) )
      avg_slowdown_l.append(sim_m['avg_slowdown'] )
      
      max_cpu_util_l.append(sim_m['max_cpu_util'] )
      avg_cpu_util_l.append(sim_m['avg_cpu_util'] )
      max_mem_util_l.append(sim_m['max_mem_util'] )
      avg_mem_util_l.append(sim_m['avg_mem_util'] )
    
    blog(drop_rate_l=drop_rate_l, evict_rate_l=evict_rate_l,
      max_cpu_util_l=max_cpu_util_l, avg_cpu_util_l=avg_cpu_util_l,
      max_mem_util_l=max_mem_util_l, avg_mem_util_l=avg_mem_util_l)
    
    if label is None:
      label = '{}'.format(sching_m['type'] )
    color, marker = next(nice_color), next(marker_cycle)
    plot.sca(axs[0] )
    plot.plot(ar_l, drop_rate_l, color=color, label=label, marker=marker, lw=2, ls=':', ms=3, mew=3)
    plot.ylim(ymin=0)
    plot.sca(axs[1] )
    plot.plot(ar_l, evict_rate_l, color=color, label=label, marker=marker, lw=2, ls=':', ms=3, mew=2)
    plot.ylim(ymin=0)
    plot.sca(axs[2] )
    plot.plot(ar_l, avg_cpu_util_l, color=color, label=label, marker=marker, lw=2, ls=':', mew=2)
    plot.ylim(ymin=0)
    plot.sca(axs[3] )
    plot.plot(ar_l, avg_mem_util_l, color=color, label=label, marker=marker, lw=2, ls=':', mew=2)
    plot.ylim(ymin=0)
  
  # plot_({'type': 'spreading', 'wrt': 'sched'}, w_prediction=True, label='w/ prediction')
  # plot_({'type': 'spreading', 'wrt': 'sched'}, w_prediction=False, label='w/o prediction')
  plot_({'type': 'packing', 'wrt': 'sched'}, w_prediction=True, label='w/ prediction')
  plot_({'type': 'packing', 'wrt': 'sched'}, w_prediction=False, label='w/o prediction')
  
  
  # print("\n*** w_dynamicreqed")
  # sinfo_m['w_dynamicreqed'] = True
  # plot_({'type': 'spreading', 'wrt': 'sched'}, w_prediction=True, label='w/ dynamic prediction')
  
  ax = axs[0]
  plot.sca(ax)
  # s = 'sinfo_m= {}'.format(sinfo_m) + \
  #     '\nar_offeredload_m= {}'.format(ar_offeredload_m)
  # an = plot.annotate(
  #   textwrap.fill(s, 150), xy=(-0.1, -0.2), xycoords='axes fraction', xytext=(-0.1, -0.2), textcoords='axes fraction', va='top')
  plot.legend()
  plot.xlabel('Arrival rate', fontsize=14)
  plot.ylabel('Pod drop rate', fontsize=14)
  ax = axs[1]
  plot.sca(ax)
  plot.legend()
  plot.xlabel('Arrival rate', fontsize=14)
  plot.ylabel('Pod eviction rate', fontsize=14)
  ax = axs[2]
  plot.sca(ax)
  plot.legend()
  plot.xlabel('Arrival rate', fontsize=14)
  plot.ylabel('Average CPU utilization', fontsize=14)
  ax = axs[3]
  plot.sca(ax)
  plot.legend()
  plot.xlabel('Arrival rate', fontsize=14)
  plot.ylabel('Average Memory utilization', fontsize=14)
  
  plot.gcf().set_size_inches(len(axs)*5, 4)
  plot.subplots_adjust(wspace=0.5)
  plot.savefig('plot_sching_wrt_requested_vs_predicted.png', bbox_inches='tight') # bbox_extra_artists=(an,), 
  log(WARNING, "done.")

'''
  For some reason the plotting in above did not work and I did not have time for debugging, 
  which resulted in this `ugly` function for manual plotting.
'''
def plot_manual():
  ar_l = [14.38, 23.96, 33.54, 43.13, 52.71, 62.3, 71.88]
  # With spreading without prediction
  drop_rate_wspreading_wopredict_l = [0.7757238915217229, 0.8251496411097037, 0.85, 0.8617493047378048, 0.8707972619287296, 0.8816689637889988, 0.8881925793295534]
  evict_rate_wspreading_wopredict_l = [2.4812048730863707e-05, 2.4836698706007996e-05, 7.496251874062968e-05, 5.010898704682685e-05, 5.033219247030401e-05, 0, 0]
  max_cpu_util_wspreading_wopredict_l = [0.11651938582625747, 0.13013711747801343, 0.1463783921862511, 0.15818977486789343, 0.17490395959915359, 0.16402021176252796, 0.1740548014931909]
  avg_cpu_util_wspreading_wopredict_l = [0.03758272095524428, 0.04740200904153237, 0.06243842650105104, 0.07387983870760464, 0.08329424011140468, 0.07996394499043716, 0.08873796447864082]
  max_mem_util_wspreading_wopredict_l = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  avg_mem_util_wspreading_wopredict_l = [0.040207444176533505, 0.04465320863848647, 0.046134668132602766, 0.04692597350745011, 0.05149382884647097, 0.049949201257058026, 0.05110955942277427]
  # With spreading with prediction
  drop_rate_wspreading_wpredict_l = [0, 0, 0, 0.0002255978342607911, 0.004422346592044773, 0.0299553538024094, 0.07257984280171127]
  evict_rate_wspreading_wpredict_l = [4.987406797835465e-05, 2.506831114787797e-05, 7.471421811570742e-05, 7.519927808693036e-05, 2.4985008994603238e-05, 4.9884019654303743e-05, 0]
  max_cpu_util_wspreading_wpredict_l = [0.34811766559429325, 0.46077883476370235, 0.5502596264656189, 0.6059647707979727, 0.6136526295125522, 0.5924291922939421, 0.6200892448154092]
  avg_cpu_util_wspreading_wpredict_l = [0.13871709615697195, 0.22588281438087873, 0.30737958704900287, 0.344172186081082, 0.37634590778577276, 0.37911288751983774, 0.3924049259004512]
  max_mem_util_wspreading_wpredict_l = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  avg_mem_util_wspreading_wpredict_l = [0.2512703221059454, 0.3732329709474165, 0.5223708647505548, 0.5885482946136623, 0.7012394543801259, 0.7816506658610525, 0.8424787588685536]
  
  # With packing without prediction
  drop_rate_wpacking_wopredict_l = [0.7766870255433842, 0.8322593536140667, 0.8511240174235217, 0.8712757448510298, 0.8899200241436548, 0.8931473863750377, 0.9078509316770186]
  evict_rate_wpacking_wopredict_l = [0, 2.4976272541085968e-05, 5.006759124818505e-05, 2.499500099980004e-05, 5.02992807202857e-05, 5.016554630279924e-05, 0]
  max_cpu_util_wpacking_wopredict_l = [0.11820961564294992, 0.12240946499581377, 0.1329432226840405, 0.14062811566758388, 0.13606314863166608, 0.1482482795206025, 0.13735948089067754]
  avg_cpu_util_wpacking_wopredict_l = [0.03561170170694197, 0.04533387708962264, 0.05832219171565252, 0.06157980728702991, 0.06462784675497137, 0.07324263854996202, 0.06636054790384821]
  max_mem_util_wpacking_wopredict_l = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  avg_mem_util_wpacking_wopredict_l = [0.040413713847422865, 0.04205081616506229, 0.04700504069315781, 0.04793903977432027, 0.04670984627514842, 0.04941914077711255, 0.048247808881800816]
  # With packing with prediction
  drop_rate_wpacking_wpredict_l = [0, 0, 0, 0.00012550515826200456, 0.011653496048814644, 0.042479160930185986, 0.08717502176887673]
  evict_rate_wpacking_wpredict_l = [0.00012584314909896304, 0.0001736714136853074, 0.00020083850074059198, 0.0002008082532192073, 0.00017505251575472642, 0.00037547873538761923, 0.0004975743251648215]
  max_cpu_util_wpacking_wpredict_l = [0.31112867681680634, 0.4303001212521738, 0.5611111021764335, 0.7700480694078454, 0.7069923226272518, 0.7148262760503818, 0.6695400474944028]
  avg_cpu_util_wpacking_wpredict_l = [0.15212062755588773, 0.23190637510828446, 0.3040243657152918, 0.4132361686026411, 0.43899244352986655, 0.42209591954461656, 0.4025502958121631]
  max_mem_util_wpacking_wpredict_l = [0.40132076263427735, 0.6, 0.7161022577285767, 0.9313971052169799, 0.9542531919479369, 1.0, 1.0]
  avg_mem_util_wpacking_wpredict_l = [0.24298734342977882, 0.3734415726644508, 0.5042272143161334, 0.6341789377586553, 0.7026009705618355, 0.780413810815117, 0.8628751314228962]
  
  def plot_(wopredict_l, wpredict_l, xlabel):
    plot.plot(ar_l, wopredict_l, color=NICE_BLUE, label='w/o prediction', marker='+', lw=3, ls=':', ms=5, mew=5)
    plot.plot(ar_l, wpredict_l, color=NICE_RED, label='w/ prediction', marker='x', lw=3, ls=':', ms=5, mew=5)
    plot.ylim(ymin=0)
    plot.legend()
    plot.xlabel('Arrival rate', fontsize=14)
    plot.ylabel(xlabel, fontsize=14)
    fig = plot.gcf()
    fig.set_size_inches(3, 2)
    plot.subplots_adjust(wspace=0.5)
    plot.savefig('plot_{}.png'.format(xlabel), bbox_inches='tight') # bbox_extra_artists=(an,), 
    fig.clear()
    log(WARNING, "done with {}.".format(xlabel) )
  
  plot_(drop_rate_wspreading_wopredict_l, drop_rate_wspreading_wpredict_l, 'Pod drop rate')
  plot_(evict_rate_wspreading_wopredict_l, evict_rate_wspreading_wpredict_l, 'Pod eviction rate')
  plot_(avg_cpu_util_wspreading_wopredict_l, avg_cpu_util_wspreading_wpredict_l, 'Average CPU\nutilization')
  plot_(avg_mem_util_wspreading_wopredict_l, avg_mem_util_wspreading_wpredict_l, 'Average Memory\nutilization')
  
  # plot_(drop_rate_wpacking_wopredict_l, drop_rate_wspreading_wpredict_l, 'Pod drop rate')
  # plot_(evict_rate_wpacking_wopredict_l, evict_rate_wspreading_wpredict_l, 'Pod eviction rate')
  # plot_(avg_cpu_util_wpacking_wopredict_l, avg_cpu_util_wspreading_wpredict_l, 'Average CPU utilization')
  # plot_(avg_mem_util_wpacking_wopredict_l, avg_mem_util_wspreading_wpredict_l, 'Average Memory utilization')

if __name__ == '__main__':
  # plot_wrt_ar()
  # plot_overbook()
  plot_sching_wrt_requested_vs_predicted()
  # plot_manual()
