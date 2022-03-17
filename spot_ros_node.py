import argparse
import time

import blosc
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from spot_wrapper.spot import Spot, image_response_to_cv2, scale_depth_img
from std_msgs.msg import ByteMultiArray, Float32MultiArray

RGB_TOPIC = "/camera/color/image_raw/compressed"
DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw/compressed"
ROBOT_STATE_TOPIC = "/robot_state"

MAX_DEPTH = 10.0
MIN_DEPTH = 0.3
FILTER_FRONT_DEPTH = False
CLAMP_DEPTH = False

class SpotRosSubscriber:
    def __init__(self, node_name):
        rospy.init_node(node_name, disable_signals=True)

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Instantiate subscribers
        rospy.Subscriber(
            DEPTH_TOPIC,
            Image,
            self.front_depth_callback,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.Subscriber(
            RGB_TOPIC,
            Image,
            self.front_rgb_callback,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.Subscriber(
            ROBOT_STATE_TOPIC,
            Float32MultiArray,
            self.robot_state_callback,
            queue_size=1,
        )

        # Msg holders
        self.front_depth = None
        self.front_rgb = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.depth_updated = False
        self.rgb_updated = False
        rospy.loginfo(f"[{node_name}]: Subscribing has started.")

    def front_depth_callback(self, msg):
        self.front_depth = msg
        self.depth_updated = True

    def front_rgb_callback(self, msg):
        self.front_rgb = msg
        self.rgb_updated = True

    def robot_state_callback(self, msg):
        self.x, self.y, self.yaw = msg.data[:3]

    @property
    def front_depth_img(self):
        if self.front_depth is None or not self.depth_updated:
            print("IMAGE IS NONE!")
            return None
        # Gather latest images
        if isinstance(self.front_depth, ByteMultiArray):
            return decode_ros_blosc(self.front_depth)
        elif isinstance(self.front_depth, CompressedImage):
            return self.cv_bridge.compressed_imgmsg_to_cv2(self.front_depth)
        elif isinstance(self.front_depth, Image):
            return self.cv_bridge.imgmsg_to_cv2(self.front_depth)

    @property
    def front_rgb_img(self):
        if self.front_rgb is None or not self.rgb_updated:
            print("IMAGE IS NONE!")
            return None
        # Gather latest images
        if isinstance(self.front_rgb, ByteMultiArray):
            return decode_ros_blosc(self.front_rgb)
        elif isinstance(self.front_rgb, CompressedImage):
            return self.cv_bridge.compressed_imgmsg_to_cv2(self.front_rgb)
        elif isinstance(self.front_rgb, Image):
            return self.cv_bridge.imgmsg_to_cv2(self.front_rgb)


class SpotRosProprioceptionPublisher:
    def __init__(self, spot, verbose=False):
        rospy.init_node("spot_ros_proprioception_node", disable_signals=True)
        self.spot = spot

        # Instantiate filtered image publishers
        self.pub = rospy.Publisher(ROBOT_STATE_TOPIC, Float32MultiArray, queue_size=1)
        self.last_publish = time.time()
        rospy.loginfo("[spot_ros_proprioception_node]: Publishing has started.")

    def publish_msgs(self):
        while time.time() - self.last_publish < 1 / 100:
            # Limit to 100 Hz max
            pass
        st = time.time()
        robot_state = self.spot.get_robot_state()
        if self.verbose:
            rospy.loginfo(
                f"[spot_ros_proprioception_node]: Proprioception retrieval / publish time: "
                f"{1/(time.time() - st):.4f} / {1/(time.time() - self.last_publish):.4f} Hz"
            )
        msg = Float32MultiArray()
        xy_yaw = self.spot.get_xy_yaw(robot_state=robot_state)
        msg.data = np.array(
            list(xy_yaw),
            dtype=np.float32,
        )
        self.pub.publish(msg)
        self.last_publish = time.time()


def decode_ros_blosc(msg: ByteMultiArray):
    byte_data = msg.data
    byte_data = [(i + 128).to_bytes(1, "big") for i in byte_data]
    decoded = blosc.unpack_array(b"".join(byte_data))
    return decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--proprioception", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.proprioception:
        name = "spot_ros_proprioception_node"
        cls = SpotRosProprioceptionPublisher
    else:
        name = "spot_ros_node"
        cls = SpotRosPublisher

    spot = Spot(name)
    srn = cls(spot, args.verbose)
    while not rospy.is_shutdown():
        srn.publish_msgs()


if __name__ == "__main__":
    main()
