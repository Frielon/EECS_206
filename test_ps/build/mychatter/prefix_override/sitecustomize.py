import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/cc/ee106a/sp26/class/ee106a-acq/ros_workspaces/test_ps/install/mychatter'
