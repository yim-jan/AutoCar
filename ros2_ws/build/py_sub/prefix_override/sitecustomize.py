import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/media/sf_Linux_Share/Robot/ros2_ws/install/py_sub'
