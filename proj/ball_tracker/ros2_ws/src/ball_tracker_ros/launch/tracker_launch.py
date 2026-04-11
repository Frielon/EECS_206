from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('camera_index', default_value='10'),
        DeclareLaunchArgument('camera_width', default_value='640'),
        DeclareLaunchArgument('camera_height', default_value='480'),
        DeclareLaunchArgument('camera_fps', default_value='60'),
        DeclareLaunchArgument('show_debug', default_value='true'),
        DeclareLaunchArgument('calibration_file', default_value=''),

        Node(
            package='ball_tracker_ros',
            executable='tracker_node',
            name='ball_tracker',
            output='screen',
            parameters=[{
                'camera_index': LaunchConfiguration('camera_index'),
                'camera_width': LaunchConfiguration('camera_width'),
                'camera_height': LaunchConfiguration('camera_height'),
                'camera_fps': LaunchConfiguration('camera_fps'),
                'show_debug': LaunchConfiguration('show_debug'),
                'calibration_file': LaunchConfiguration('calibration_file'),
            }],
        ),
    ])
