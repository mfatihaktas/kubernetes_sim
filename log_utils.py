"""
A simple, unified logging that can be refactored easily
(I am aware of logging module and I don't find it flexible enough ...)
"""
import inspect, pprint

INFO = 0
DEBUG = 1
WARNING = 2
ERROR = 3

debug_level__string_map = {INFO: "INFO", DEBUG: "DEBUG", WARNING: "WARNING", ERROR: "ERROR"}

# SIM_DEBUG_LEVEL = INFO
SIM_DEBUG_LEVEL = WARNING
# SIM_DEBUG_LEVEL = ERROR

def slog(dlevel, env, caller, action, affected, **kwargs):
  """
  Parameters
  ----------
  dlevel= int -- debug level
  env= simpy.Environment
  caller= string -- name of the sim component acting
  action= string
  affected= any -- whatever component being acted on/with e.g., packet
  """
  if SIM_DEBUG_LEVEL <= dlevel:
    print("{} t: {:.2f}] {} {}\n\t{}".format(debug_level__string_map[dlevel], env.now, caller, action, affected) )
    for k, v in kwargs.items():
      print("  {}: {}".format(k, pprint.pformat(v) ) )

DEBUG_LEVEL = INFO
# DEBUG_LEVEL = WARNING
# DEBUG_LEVEL = ERROR

def log(dlevel, log, **kwargs):
  """
  Parameters
  ----------
  dlevel= int -- debug level
  log= string to be logged
  """
  try:
    funcname = inspect.stack()[1][3]
  except IndexError:
    funcname = ''
  
  if DEBUG_LEVEL <= dlevel:
    print("{}] {}:: {}".format(debug_level__string_map[dlevel], funcname, log) )
    for k, v in kwargs.items():
      print("  {}: {}".format(k, pprint.pformat(v) ) )

def blog(**kwargs):
  for k, v in kwargs.items():
    print("  {}: {}".format(k, pprint.pformat(v) ) )
