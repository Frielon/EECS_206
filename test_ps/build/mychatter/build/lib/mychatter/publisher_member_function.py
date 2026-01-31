# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# These lines allow us to import rclpy so we can use Python and its Node class
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import threading
# This line imports the built-in string message type that our node will use to structure its data to pass on our topic
# from mychatter.msg import mydata

# We're creating a class called Talker, which is a subclass of Node
class MinimalPublisher(Node):

    # Here, we define the constructor
    def __init__(self):
        # We call the Node class's constructor and call it "minimal_publisher"
        super().__init__('minimal_publisher')
        
         # Here, we set that the node publishes message of type String (where did this type come from?), over a topic called "chatter_talk", and with queue size 10. The queue size limits the amount of queued messages if a subscriber doesn't receive them quickly enough.
        self.publisher_ = self.create_publisher(String, 'chatter_talk', 10)
        
        # We create a timer with a callback (a function that runs automatically when something happens so you don't have to constantly check if something has happened) 
        timer_period = 0.5  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
    
    # Here we create a message with the counter value appended and publish it
    def timer_callback(self,name):
        # msg = {'name': '0', "stamp"}


        # msg.name = 'name'

        this_time = self.get_clock().now().to_msg().nanosec

        print(this_time)

        mydata = {"name": name, "stamp": this_time, "status": 'true'}

        data_str = json.dumps(mydata)

        msg = String()
        msg.data = data_str
        # msg.stamp = this_time
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: name-{name}, time-{this_time}')
        self.i += 1


def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
    # Create the node
    minimal_publisher = MinimalPublisher()
    # Spin the node so its callbacks are called
    thread = threading.Thread(target=rclpy.spin, args=(minimal_publisher,), daemon = True)
    # rclpy.spin(minimal_publisher)
    thread.start()

    try: 
        while rclpy.ok():
            name = input('enter something')
            minimal_publisher.timer_callback(name)
    except KeyboardInterrupt:
        pass

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
