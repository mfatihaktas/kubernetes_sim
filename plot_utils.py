import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import itertools, scipy #, seaborn
import numpy as np

NICE_BLUE = '#66b3ff'
NICE_RED = '#ff9999'
NICE_GREEN = '#99ff99'
NICE_ORANGE = '#ffcc99'
NICE_PURPLE = 'mediumpurple'

nice_color = itertools.cycle((NICE_BLUE, NICE_RED, NICE_GREEN, NICE_ORANGE))
dark_color = itertools.cycle(('green', 'purple', 'blue', 'magenta', 'purple', 'gray', 'brown', 'turquoise', 'gold', 'olive', 'silver', 'rosybrown', 'plum', 'goldenrod', 'lightsteelblue', 'lightpink', 'orange', 'darkgray', 'orangered'))
light_color = itertools.cycle(('silver', 'rosybrown', 'plum', 'lightsteelblue', 'lightpink', 'orange', 'turquoise'))
linestyle = itertools.cycle(('-', '--', '-.', ':') )
marker_cycle = itertools.cycle(('x', '+', 'v', '^', 'p', 'd', '<', '>', '1' , '2', '3', '4') )
skinny_marker_l = ['x', '+', '1', '2', '3', '4']

mew, ms = 1, 2 # 3, 5

def prettify(ax):
  ax.patch.set_alpha(0.2)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

def area_under_cdf(l, xl, xu):
  x_l = sorted(l)
  y_l = np.arange(len(x_l) )/len(x_l)
  
  il = 0
  while x_l[il] < xl: il += 1
  iu = 0
  while x_l[iu] < xu: iu += 1
  return np.trapz(y=y_l[il:iu], x=x_l[il:iu] )

def avg_within(x_l, xl, xu):
  return np.mean([x for x in x_l if x >= xl and x <= xu] )

def CDFval_atx_l(l, atx_l):
  x_l = sorted(l)
  y_l = np.arange(len(x_l) )/len(x_l)
  
  def val_atx(x):
    i = 0
    while x_l[i] < x: i += 1
    return y_l[i]
  
  return {x: val_atx(x) for x in atx_l}

def add_pdf(l, label, color, bins=50):
  # w_l = np.ones_like(l)/float(len(l) )
  # plot.hist(l, bins=bins, weights=w_l, label=label, color=color, edgecolor='black')
  
  # n = len(l)//bins
  # p, x = np.histogram(l, bins=n) # bin it into n = N//10 bins
  # x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers
  # f = scipy.interpolate.UnivariateSpline(x, p, s=n)
  # plot.plot(x, f(x), label=label, color=color, ls='--', lw=2, mew=2, ms=2)
  
  # density = scipy.stats.gaussian_kde(l)
  # # xs = np.linspace(0, 8, 200)
  # density.covariance_factor = lambda : .25
  # density._compute_covariance()
  # plot.plot(l, density(l) )
  
  seaborn.distplot(l, hist=False, norm_hist=True, kde=True, bins=bins, label=label, color=color,
    hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 3} )

def add_cdf(l, ax, label, color, drawline_x_l=[] ):
  plot.sca(ax)
  x_l = sorted(l)
  y_l = np.arange(len(x_l) )/len(x_l)
  plot.plot(x_l, y_l, label=label, color=color, marker='.', linestyle=':', lw=2, mew=2, ms=2) # lw=1, mew=1, ms=1
  
  def drawline(x):
    i = 0
    while i < len(x_l) and x_l[i] < x: i += 1
    if i == len(x_l):
      return
    ax.add_line(
      matplotlib.lines.Line2D([x_l[i], x_l[i]], [0, y_l[i]], color=color, linestyle='--') )
    ax.add_line(
      matplotlib.lines.Line2D([0, x_l[i]], [y_l[i], y_l[i]], color=color, linestyle='--') )
  
  for x in drawline_x_l:
    drawline(x)

