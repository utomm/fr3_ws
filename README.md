# fr3_ws

A ROS package for [collecting realworld data and train a minidiffuser in real world].  
Tested on **ROS 1** and we are working on a ROS2 verison

## Prerequisites

- Ubuntu with ROS1 [ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu)
- [catkin tools](https://catkin-tools.readthedocs.io/en/latest/installing.html) (optional but recommended)
- Franka related libraries, see [franka with moveit](https://github.com/utomm/fr3_moveit_config)

## Installation

1. **Clone this repository** into your ROS workspace's `src` directory:

    ```bash
    cd ~/catkin_ws/src
    git clone https://github.com/utomm/fr3_ws.git
    ```

2. **Install dependencies** (if any, e.g., using `rosdep`):

    ```bash
    cd ~/catkin_ws
    rosdep install --from-paths src --ignore-src -r -y
    ```

3. **Compile the workspace**:

    ```bash
    cd ~/catkin_ws
    catkin_make        # or: catkin build
    ```

4. **Source the workspace**:

    ```bash
    source devel/setup.bash
    ```

## Usage

![image](https://github.com/user-attachments/assets/782dcc30-1946-4fb7-828e-4c47a35311a7)


Basically we use another interface of websocket to communicate between the Learning method env and the ROS env. To aviod dealing with ROS python verison requirement.

1. **Data collection**

   on Minidiffuser side

    ```bash
    python collection_client
    ```

    on ROS side, open two terminal side

    ```bash
    rosrun fr3_ws collection_server
    ```

    this bring up the ros-to-websocket convert server

    ```bash
    roslaunch fr3_ws collect_setup.launch
    ```

    this bring up the moveit and rviz interface to control the robot and collect keypose, you can record the keypose using moveit planner, or using franka's inbuilt inverser kinematice mode.

2. **Train model using collected dataset**

   on Minidiffuser side

   `python preprocess/realworld_downsample.py`

   and then
   
   `python minidiffuser/train/train_diffusion_realworld.py`
   

4. **Run a trained policy**

   on Minidiffuser side

    ```bash
    python realworld_env.py
    ```

    on ROS side, open two terminal side

    ```bash
    rosrun fr3_ws env_server.py
    ```

    this bring up the ros-to-websocket convert server

    ```bash
    roslaunch fr3_ws collect_setup.launch
    ```

