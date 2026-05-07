#!/usr/bin/env python3
"""
Standalone AI Gesture Control for TurtleBot3 Waffle Pi
完全在TurtleBot3上运行的AI手势控制系统

This node runs ENTIRELY on TurtleBot3 using its onboard camera.
No workstation needed!

Features:
- MediaPipe hand gesture recognition
- Hybrid navigation (Rule-based 70% + RL 30%)
- All processing on TurtleBot3
- Uses TurtleBot3's camera

Author: Your Name
Date: 2026-05-07
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import mediapipe as mp
import numpy as np
import time

class StandaloneTurtleBotGestureControl(Node):
    """
    All-in-one node: gesture recognition + robot control + hybrid navigation
    单节点：手势识别 + 机器人控制 + 混合导航
    """
    
    def __init__(self):
        super().__init__('standalone_gesture_control')
        
        # ROS2 publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # MediaPipe hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Camera setup (TurtleBot3's camera)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Hybrid navigation parameters
        self.rule_weight = 0.7  # 70% rule-based
        self.rl_weight = 0.3    # 30% RL-based
        
        # Safety parameters
        self.max_linear_speed = 0.22  # m/s
        self.max_angular_speed = 2.0  # rad/s
        self.last_command_time = time.time()
        self.command_timeout = 1.0  # Stop if no gesture for 1 second
        
        # Current state
        self.current_gesture = "none"
        self.gesture_history = []
        self.history_size = 5
        
        # RL Q-table (simplified for onboard processing)
        self.q_table = self._initialize_q_table()
        
        # Create timer for main loop
        self.timer = self.create_timer(0.1, self.process_frame)  # 10 Hz
        
        self.get_logger().info('Standalone Gesture Control Started!')
        self.get_logger().info('Using TurtleBot3 onboard camera')
        self.get_logger().info('All processing on TurtleBot3 - No workstation needed!')
        
    def _initialize_q_table(self):
        """Initialize simplified Q-table for RL component"""
        # States: gestures (forward, backward, left, right, stop)
        # Actions: velocity adjustments
        q_table = {}
        gestures = ['forward', 'backward', 'left', 'right', 'stop']
        actions = ['increase', 'decrease', 'maintain']
        
        for gesture in gestures:
            q_table[gesture] = {}
            for action in actions:
                q_table[gesture][action] = np.random.uniform(-0.1, 0.1)
        
        return q_table
    
    def recognize_gesture(self, hand_landmarks):
        """
        Recognize hand gesture from landmarks
        从手部关键点识别手势
        
        Gestures:
        - Index finger up: Forward
        - Index finger down: Backward  
        - Thumb left: Turn left
        - Thumb right: Turn right
        - Open palm: Stop
        """
        if hand_landmarks is None:
            return "none"
        
        # Get key landmarks
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        wrist = hand_landmarks.landmark[0]
        index_mcp = hand_landmarks.landmark[5]
        
        # Calculate if fingers are extended
        thumb_extended = thumb_tip.x < index_mcp.x - 0.05
        thumb_right = thumb_tip.x > index_mcp.x + 0.05
        index_extended = index_tip.y < index_mcp.y
        middle_extended = middle_tip.y < hand_landmarks.landmark[9].y
        ring_extended = ring_tip.y < hand_landmarks.landmark[13].y
        pinky_extended = pinky_tip.y < hand_landmarks.landmark[17].y
        
        # Open palm (all fingers extended)
        if (index_extended and middle_extended and 
            ring_extended and pinky_extended):
            return "stop"
        
        # Index finger pointing up (forward)
        if index_extended and not middle_extended:
            if index_tip.y < wrist.y - 0.1:
                return "forward"
        
        # Index finger pointing down (backward)
        if not index_extended:
            if index_tip.y > wrist.y + 0.1:
                return "backward"
        
        # Thumb left (turn left)
        if thumb_extended and not index_extended:
            return "left"
        
        # Thumb right (turn right)
        if thumb_right and not index_extended:
            return "right"
        
        return "none"
    
    def get_rule_based_velocity(self, gesture):
        """
        Rule-based velocity calculation (70% of hybrid)
        基于规则的速度计算（混合导航的70%）
        """
        twist = Twist()
        
        if gesture == "forward":
            twist.linear.x = self.max_linear_speed
            twist.angular.z = 0.0
        elif gesture == "backward":
            twist.linear.x = -self.max_linear_speed * 0.5
            twist.angular.z = 0.0
        elif gesture == "left":
            twist.linear.x = 0.0
            twist.angular.z = self.max_angular_speed * 0.5
        elif gesture == "right":
            twist.linear.x = 0.0
            twist.angular.z = -self.max_angular_speed * 0.5
        elif gesture == "stop":
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        
        return twist
    
    def get_rl_velocity_adjustment(self, gesture):
        """
        RL-based velocity adjustment (30% of hybrid)
        基于强化学习的速度调整（混合导航的30%）
        """
        if gesture not in self.q_table:
            return 0.0, 0.0
        
        # Get best action from Q-table
        actions = self.q_table[gesture]
        best_action = max(actions, key=actions.get)
        
        # Convert action to velocity adjustment
        linear_adj = 0.0
        angular_adj = 0.0
        
        if best_action == 'increase':
            linear_adj = 0.05
            angular_adj = 0.1
        elif best_action == 'decrease':
            linear_adj = -0.05
            angular_adj = -0.1
        
        return linear_adj, angular_adj
    
    def hybrid_navigation(self, gesture):
        """
        Hybrid navigation: 70% rule-based + 30% RL
        混合导航：70%规则 + 30%强化学习
        """
        # Get rule-based velocity
        rule_vel = self.get_rule_based_velocity(gesture)
        
        # Get RL adjustment
        rl_linear_adj, rl_angular_adj = self.get_rl_velocity_adjustment(gesture)
        
        # Combine with weights
        final_twist = Twist()
        final_twist.linear.x = (self.rule_weight * rule_vel.linear.x + 
                               self.rl_weight * rl_linear_adj)
        final_twist.angular.z = (self.rule_weight * rule_vel.angular.z + 
                                self.rl_weight * rl_angular_adj)
        
        # Apply safety limits
        final_twist.linear.x = np.clip(final_twist.linear.x, 
                                       -self.max_linear_speed, 
                                       self.max_linear_speed)
        final_twist.angular.z = np.clip(final_twist.angular.z,
                                        -self.max_angular_speed,
                                        self.max_angular_speed)
        
        return final_twist
    
    def smooth_gesture(self, new_gesture):
        """
        Smooth gesture recognition with history
        使用历史记录平滑手势识别
        """
        self.gesture_history.append(new_gesture)
        
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
        
        # Most common gesture in history
        if len(self.gesture_history) >= 3:
            from collections import Counter
            gesture_counts = Counter(self.gesture_history)
            smoothed_gesture = gesture_counts.most_common(1)[0][0]
            return smoothed_gesture
        
        return new_gesture
    
    def process_frame(self):
        """
        Main processing loop: capture frame, detect gesture, control robot
        主处理循环：捕获帧、检测手势、控制机器人
        """
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().warn('Failed to capture frame')
            return
        
        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        gesture = "none"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Recognize gesture
                gesture = self.recognize_gesture(hand_landmarks)
        
        # Smooth gesture
        smoothed_gesture = self.smooth_gesture(gesture)
        
        # Update current gesture
        if smoothed_gesture != "none":
            self.current_gesture = smoothed_gesture
            self.last_command_time = time.time()
        
        # Check timeout
        if time.time() - self.last_command_time > self.command_timeout:
            self.current_gesture = "stop"
        
        # Calculate velocity using hybrid navigation
        cmd_vel = self.hybrid_navigation(self.current_gesture)
        
        # Publish velocity command
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Display info on frame
        self._draw_info(frame, smoothed_gesture, cmd_vel)
        
        # Show frame
        cv2.imshow('TurtleBot3 Gesture Control', frame)
        cv2.waitKey(1)
    
    def _draw_info(self, frame, gesture, cmd_vel):
        """Draw information on frame"""
        # Draw gesture
        cv2.putText(frame, f'Gesture: {gesture}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw velocity
        cv2.putText(frame, f'Linear: {cmd_vel.linear.x:.2f} m/s', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'Angular: {cmd_vel.angular.z:.2f} rad/s', (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw hybrid info
        cv2.putText(frame, 'Hybrid: 70% Rule + 30% RL', (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw instructions
        instructions = [
            "Point up: Forward",
            "Point down: Backward",
            "Thumb left: Turn left",
            "Thumb right: Turn right",
            "Open palm: Stop"
        ]
        
        y_offset = 200
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
    
    def destroy_node(self):
        """Cleanup when shutting down"""
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = StandaloneTurtleBotGestureControl()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
