import os, json, re, errno, scipy
import numpy as np

from log_utils import *

def gen_allfnames_indir(dirurl, extension='.json', prefix=None):
  rx = re.compile(r'\{}'.format(extension) )
  l = []
  for path, dnames, fnames in os.walk(dirurl):
    for f in fnames:
      if rx.search(f):
        if prefix is None:
          yield f
        elif f.startswith(prefix):
          yield f
          

def gen_allfurls_indir(dirurl, extension='.json', prefix=None):
  rx = re.compile(r'\{}'.format(extension) )
  l = []
  for path, dnames, fnames in os.walk(dirurl):
    for f in fnames:
      if rx.search(f):
        if prefix is None:
          yield os.path.join(path, f)
        elif f.startswith(prefix):
          yield os.path.join(path, f)

def read_fromfurl(furl):
  try:
    with open(furl, "r") as fp:
      return fp.read()
  except:
    log(ERROR, 'in fp.read;', furl=furl, error_info=sys.exc_info()[0] )
    return -1

def write_tofurl(data, furl):
  if not os.path.exists(os.path.dirname(furl) ):
    try:
      os.makedirs(os.path.dirname(furl) )
    except OSError as exc: # Guard against race condition
      if exc.errno != errno.EEXIST:
        log(ERROR, "exception;", errno=exc.errno)
        raise
  try:
    with open(furl, "w") as fp:
      fp.write(data)
  except:
    log(ERROR, 'in fp.write;', furl=furl, error_info=sys.exc_info()[0] )
    return 1

def writejson_tofurl(data, furl):
  if not os.path.exists(os.path.dirname(furl) ):
    try:
      os.makedirs(os.path.dirname(furl) )
    except OSError as exc: # Guard against race condition
      if exc.errno != errno.EEXIST:
        raise
  
  with open(furl, "w") as fp:
    json.dump(data, fp, indent=2, sort_keys=True)

def read_strip_jsonf(furl):
  try:
    with open(furl) as f:
      return f.read().replace('\n','').replace(" ", "")
  except:
    log(ERROR, "file does not exist!;", furl=furl)
    return -1

def is_nan(e):
  try:
    if np.isnan(e):
      return True
  except:
    return False

def is_any_nan(l):
  for e in l:
    if is_nan(e):
      return True
  return False

def is_any_if(l, f_l):
  for e in l:
    for f in f_l:
      if f(e):
        return True
  return False

def append_if(l, e, f_l):
  for f in f_l:
    if not f(e):
      return
  l.append(e)

def shorten_podname(pod):
  return pod[:40]+'...' if len(pod) > 40 else pod

def normed_correlation(x_l, y_l):
  if len(x_l) == 0 or len(y_l) == 0:
    return np.nan
  x_sqrsum = math.sqrt(sum([x**2 for x in x_l] ) )
  y_sqrsum = math.sqrt(sum([y**2 for y in y_l] ) )
  if x_sqrsum == 0 or y_sqrsum == 0:
    return 0
  return max(np.correlate(x_l, y_l, 'full') )/x_sqrsum/y_sqrsum

def get_podtype(podname):
  '''
  Check if there is digits in s
  :param s: Input string
  :return: True if s has digits
  '''
  num_there = lambda s: any(i.isdigit() for i in s)
  
  '''
  Get the pod name by their common prefixes
  :param podname: Pod name
  :return: The common prefix of pods
  '''
  # l = ['tr', 'learner', 'lhelper', 's3fs-copy-driver-pog', 'armada-sre-onboard', 'ibm-keepalived', \
  #     'ibmcloud-object-storage', 'emotion-analysis-tone', 'ibm-kube-fluentd', 'kube-system.calico-node']
  l = ['ibm-system.armada-sre-onboard',
       'kube-system.ibmcloud-object-storage-driver',
       'kube-system.ibm-kube-fluentd',
       'kube-system.ibm-keepalived-watcher',
       'kube-system.calico-node']
  # l = []
  for e in l:
    if e in podname:
      return e
  
  if "-" not in podname:
    podprefix = "".join(["#" if char.isdigit() else char for char in podname])
    return podprefix
  
  podname_parts = podname.split("-")
  podprefix = podname_parts[0]
  
  # Replace the possible digits in the first part to "x"
  podprefix = "".join(["#" if char.isdigit() else char for char in podprefix])
  
  for p_part in podname_parts[1:]:
    if re.match('v[0-9]', p_part):
      podprefix += "-" + p_part
    elif num_there(p_part):
      return podprefix
    elif re.match('[a-z]+', p_part):
      podprefix += "-" + p_part
    else:
      return podprefix
  
  return podprefix

def do_distributions_differ(v1_l, v2_l):
  # Compare with t-test
  # teststatistic, pvalue = scipy.stats.ttest_ind(v1_l, v2_l, equal_var=False)
  # Compare with Kolmogorov-Smirnov test
  teststatistic, pvalue = scipy.stats.ks_2samp(v1_l, v2_l)
  # print("teststatistic= {}, pvalue= {}".format(teststatistic, pvalue) )
  # print("np.mean(v1_l)= {}, max(v1_l)= {}".format(np.mean(v1_l), max(v1_l) ) )
  # print("np.mean(v2_l)= {}, max(v2_l)= {}".format(np.mean(v2_l), max(v2_l) ) )
  if pvalue < 0.05: # distributions differ
    return True
  else:
    return False

def map_to_key__val_l(m):
  k_l, v_l = [], []
  for k, v in m.items():
    k_l.append(k)
    v_l.append(v)
  return k_l, v_l

if __name__ == '__main__':
  # cleanwritemetrics_fromdir_todir('/metrics/watson-prdwat-dal10-cruiser5', '/metrics/prdwat-dal10-cruiser5')
  log(INFO, "done.")
