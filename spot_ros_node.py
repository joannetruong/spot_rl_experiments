import argparse
import time

import blosc
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2, scale_depth_img
from std_msgs.msg import ByteMultiArray, Float32MultiArray

FRONT_DEPTH_TOPIC = "/spot_cams/filtered_front_depth"
ROBOT_STATE_TOPIC = "/robot_state"
SRC2MSG = {
    SpotCamIds.FRONTLEFT_DEPTH: ByteMultiArray,
    SpotCamIds.FRONTRIGHT_DEPTH: ByteMultiArray,
}
MAX_DEPTH = 3.5

class SpotRosPublisher:
    def __init__(self, spot):
        rospy.init_node("spot_ros_node", disable_signals=True)
        self.spot = spot

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Instantiate raw image publishers
        self.sources = list(SRC2MSG.keys())
        self.img_pubs = [
            rospy.Publisher(f"/spot_cams/{k}", v, queue_size=1)
            for k, v in SRC2MSG.items()
        ]

        # Instantiate filtered image publishers
        self.filter_front_depth = (
            SpotCamIds.FRONTLEFT_DEPTH in self.sources
            and SpotCamIds.FRONTRIGHT_DEPTH in self.sources
        )
        if self.filter_front_depth:
            self.filtered_front_depth_pub = rospy.Publisher(
                FRONT_DEPTH_TOPIC, ByteMultiArray, queue_size=1
            )

        self.last_publish = time.time()
        rospy.loginfo("[spot_ros_node]: Publishing has started.")

    def publish_msgs(self):
        st = time.time()
        image_responses = self.spot.get_image_responses(self.sources, quality=100)
        retrieval_time = time.time() - st
        # Publish raw images
        depth_eyes = {}
        for pub, src, response in zip(self.img_pubs, self.sources, image_responses):
            img = image_response_to_cv2(response)

            # Publish filtered front depth images later
            if (
                src in [SpotCamIds.FRONTRIGHT_DEPTH, SpotCamIds.FRONTLEFT_DEPTH]
                and self.filter_front_depth
            ):
                depth_eyes[src] = img

        # Filter and publish
        if self.filter_front_depth:
            # Merge
            d_keys = [SpotCamIds.FRONTRIGHT_DEPTH, SpotCamIds.FRONTLEFT_DEPTH]
            merged = np.hstack([depth_eyes[d] for d in d_keys])
            # Filter
            merged = self.filter_depth(merged, MAX_DEPTH)
            msg = blosc.pack_array(
                merged, cname="zstd", clevel=1, shuffle=blosc.NOSHUFFLE
            )
            msg = ByteMultiArray(data=[i - 128 for i in msg])
            self.filtered_front_depth_pub.publish(msg)

        rospy.loginfo(
            f"[spot_ros_node]: Image retrieval / publish time: "
            f"{1 / retrieval_time:.4f} / {1 / (time.time() - self.last_publish):.4f} Hz"
        )
        self.last_publish = time.time()

    @staticmethod
    def filter_depth(img, max_depth):
        img = scale_depth_img(img, max_depth=max_depth)
        img = np.uint8(img * 255.0)
        # Blur
        for _ in range(5):
            filtered = cv2.medianBlur(img, 19)
            filtered[img > 0] = img[img > 0]
            img = filtered

        return img


class SpotRosSubscriber:
    def __init__(self, node_name):
        rospy.init_node(node_name, disable_signals=True)

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Instantiate subscribers
        rospy.Subscriber(
            FRONT_DEPTH_TOPIC,
            ByteMultiArray,
            self.front_depth_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )
        rospy.Subscriber(
            ROBOT_STATE_TOPIC,
            Float32MultiArray,
            self.robot_state_callback,
            queue_size=1,
        )

        # Conversion between CompressedImage and cv2
        self.cv_bridge = CvBridge()

        # Msg holders
        self.front_depth = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.updated = False
        rospy.loginfo(f"[{node_name}]: Subscribing has started.")

    def front_depth_callback(self, msg):
        self.front_depth = msg
        self.updated = True

    def robot_state_callback(self, msg):
        self.x, self.y, self.yaw = msg.data[:3]

    @property
    def front_depth_img(self):
        if self.front_depth is None:
            return None
        return decode_ros_blosc(self.front_depth)


class SpotRosProprioceptionPublisher:
    def __init__(self, spot):
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
    args = parser.parse_args()

    if args.proprioception:
        name = "spot_ros_proprioception_node"
        cls = SpotRosProprioceptionPublisher
    else:
        name = "spot_ros_node"
        cls = SpotRosPublisher

    spot = Spot(name)
    srn = cls(spot)
    while not rospy.is_shutdown():
        srn.publish_msgs()


if __name__ == "__main__":
    main()
