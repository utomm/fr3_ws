#!/usr/bin/env python

import rospy
import sys
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg


def move_to_joint_angles():
    """
    Initializes MoveIt and moves the robot arm to a predefined joint configuration.
    """
    # Initialize ROS and MoveIt commander
    print("==== Initializing MoveIt! ====")
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_to_joint_angles_node", anonymous=True)

    # Instantiate a RobotCommander object. This object is the outer-level interface to
    # the robot:
    robot = moveit_commander.RobotCommander()

    # Instantiate a PlanningSceneInterface object. This object is an interface to the world
    # surrounding the robot:
    scene = moveit_commander.PlanningSceneInterface()

    # Instantiate a MoveGroupCommander object. This object is an interface to one group of joints.
    # For the FR3 robot, the planning group for the arm is typically named 'fr3_arm'.
    # You might need to check your specific MoveIt configuration for the exact group name.
    group_name = "fr3_hand"  # <-- **VERIFY THIS GROUP NAME**
    move_group = moveit_commander.MoveGroupCommander(group_name)

    # Create a DisplayTrajectory publisher which is used later to publish trajectories for RViz to visualize:
    display_trajectory_publisher = rospy.Publisher(
        "/move_group/display_planned_path",
        moveit_msgs.msg.DisplayTrajectory,
        queue_size=20,
    )

    # Get basic information about the robot
    print("==== Robot Planning Frame: %s ====" % move_group.get_planning_frame())
    print("==== End Effector link: %s ====" % move_group.get_end_effector_link())
    print("==== Available Planning Groups: %s ====" % robot.get_group_names())
    print("==== Printing robot state ====")
    print(robot.get_current_state())
    print("================================")

    # --- Define the target joint angles ---
    # Replace these values with the desired joint angles for your robot arm.
    # The order of the angles should match the order of the joints in your planning group.
    # You can get the current joint values using move_group.get_current_joint_values()
    # to help you define a target.
    # For example, if your robot has 7 joints, you might have it in degree
    # angles like this:
    # [0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0, 0.0, 0.0]
    # target_joint_angles = [
    #     0.0,
    #     -0.785,
    #     0.0,
    #     -2.356,
    #     0.0,
    #     1.571,
    #     0.785,
    # ]  # Example angles (in radians)
    target_joint_angles = [0.04, 0.04]

    # Ensure the number of target angles matches the number of joints in the group
    # if len(target_joint_angles) != len(move_group.get_joints()):
    #     rospy.logerr(
    #         "Number of target joint angles does not match the number of joints in the group '%s'",
    #         group_name,
    #     )
    #     print("we actually have", len(move_group.get_joints()), "joints")
    #     return

    # check move group
    print("Joints in fr3_arm group:", move_group.get_joints())
    print("Number of joints in fr3_arm group:", len(move_group.get_joints()))
    # print current joint values
    current_joint_values = move_group.get_current_joint_values()
    print("Current joint values:", current_joint_values)

    # --- Set the target joint values ---
    print("==== Setting target joint angles ====")
    move_group.set_joint_value_target(target_joint_angles)

    # --- Plan and execute the movement ---
    print("==== Planning to move to target joint angles ====")
    plan = move_group.go(wait=True)

    # Calling `stop()` ensures that there is no residual movement
    move_group.stop()

    # It is always good to clear your targets after planning with a group
    move_group.clear_pose_targets()

    print("==== Movement to target joint angles finished ====")

    # Shutdown MoveIt commander
    moveit_commander.roscpp_shutdown()
    print("==== MoveIt shutdown ====")


if __name__ == "__main__":
    try:
        move_to_joint_angles()
    except rospy.ROSInterruptException:
        pass
