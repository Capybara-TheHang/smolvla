from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('NTH_POC2_2_URDF_20251119v1')
    urdf_file = os.path.join(pkg_share, 'north_poc2_2_v3_1.urdf')

    return LaunchDescription([
        Node(
            package='gazebo_ros',
            executable='gazebo',
            output='screen',
            arguments=['-s', 'libgazebo_ros_factory.so'],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint'],
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-file', urdf_file, '-entity', 'NTH_POC2_2_URDF_20251119v1'],
            output='screen',
        ),
    ])
