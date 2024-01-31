from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # 获取包的共享目录
    # pkg_dir = get_package_share_directory('runec')

    # 构建 YAML 文件的路径
    # config_file_path = os.path.join(pkg_dir, 'launch', 'config.yaml')
    pkg_dir = "/workspace/src/runec"
    config_file_path = os.path.join(pkg_dir, 'launch', 'config.yaml')
    return LaunchDescription(
        [
            Node(
                package='runec',
                executable='testImageSubscriber',
                name='testImageSubscriber',
                output='screen',
                parameters=[config_file_path],
            ),
        ]
    )
