from rich import print
from .params import IK_CONFIG_ROOT, ASSET_ROOT, ROBOT_XML_DICT, IK_CONFIG_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from .motion_retarget import GeneralMotionRetargeting
from .robot_motion_viewer import RobotMotionViewer, draw_frame
from .data_loader import load_robot_motion
from .robot_to_robot_retarget import RobotBodyFrameExtractor, RobotToRobotRetargeting

from .neck_retarget import human_head_to_robot_neck

try:
    from .kinematics_model import KinematicsModel
except ImportError:
    print("Torch is not installed. Please install torch to use KinematicsModel.")
    KinematicsModel = None

try:
    from .xrobot_utils import XRobotStreamer, XRobotRecorder
except ImportError:
    print("XRobotStreamer is not installed. Please install xrobotoolkit_sdk to use this feature.")
    XRobotStreamer = None
    XRobotRecorder = None
