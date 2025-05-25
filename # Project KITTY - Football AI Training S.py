# Project KITTY - Football AI Training System
# A comprehensive AI-powered football training analysis system

import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import json
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import urllib.request
from pytube import YouTube
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FootballVideoScraper:
    """
    Part 1: Data Scraping and Preprocessing for Football Videos
    """
    
    def __init__(self, output_dir: str = "football_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Football-specific search queries
        self.search_queries = [
            "football shooting technique",
            "soccer free kick tutorial",
            "football dribbling skills",
            "soccer passing technique", 
            "football goalkeeper saves",
            "soccer penalty kicks",
            "football headers training",
            "soccer crossing technique"
        ]
    
    def extract_keywords_from_query(self, query: str) -> List[str]:
        """Extract football-specific keywords for better video filtering"""
        football_keywords = [
            'shoot', 'shooting', 'goal', 'penalty', 'free kick', 'corner',
            'pass', 'passing', 'cross', 'crossing', 'dribble', 'dribbling',
            'header', 'heading', 'tackle', 'tackling', 'save', 'goalkeeper',
            'keeper', 'defense', 'attack', 'technique', 'skill', 'training'
        ]
        
        keywords = []
        query_lower = query.lower()
        for keyword in football_keywords:
            if keyword in query_lower:
                keywords.append(keyword)
        
        return keywords if keywords else ['football', 'soccer']
    
    def download_video_sample(self, url: str, filename: str) -> bool:
        """Download video from URL (placeholder for actual implementation)"""
        try:
            # Note: In real implementation, you'd use pytube or similar
            logger.info(f"Would download video from {url} as {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            return False
    
    def standardize_video(self, input_path: str, output_path: str, 
                         target_resolution: Tuple[int, int] = (640, 480),
                         target_fps: int = 30) -> bool:
        """Standardize video format, resolution, and frame rate"""
        try:
            cap = cv2.VideoCapture(input_path)
            
            # Get original video properties
            original_fps = int(cap.get(cv2.CAP_PROP_FPS))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, target_fps, target_resolution)
            
            frame_skip = max(1, original_fps // target_fps)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    # Resize frame
                    resized_frame = cv2.resize(frame, target_resolution)
                    out.write(resized_frame)
                
                frame_count += 1
            
            cap.release()
            out.release()
            
            logger.info(f"Standardized video saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to standardize video: {e}")
            return False

class FootballPoseExtractor:
    """
    Part 2: Keypoint Extraction for Football-Specific Events
    """
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Football-specific joint indices (MediaPipe format)
        self.football_joints = {
            'shooting': [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],  # shoulders, arms, hips, legs
            'passing': [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
            'dribbling': [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],  # hips, legs, feet
            'heading': [0, 1, 2, 3, 4, 5, 11, 12, 13, 14],  # head, shoulders, arms
            'goalkeeper': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # full upper body
        }
    
    def extract_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract pose keypoints from a single frame"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                return np.array(landmarks)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract keypoints: {e}")
            return None
    
    def detect_football_action(self, keypoints_sequence: List[np.ndarray], 
                             action_type: str = 'shooting') -> Dict:
        """Detect specific football actions from keypoint sequences"""
        if not keypoints_sequence:
            return {'action': 'none', 'confidence': 0.0}
        
        # Simple heuristic-based action detection
        # In a real system, you'd use trained ML models
        
        if action_type == 'shooting':
            return self._detect_shooting_action(keypoints_sequence)
        elif action_type == 'passing':
            return self._detect_passing_action(keypoints_sequence)
        elif action_type == 'dribbling':
            return self._detect_dribbling_action(keypoints_sequence)
        elif action_type == 'heading':
            return self._detect_heading_action(keypoints_sequence)
        else:
            return {'action': 'unknown', 'confidence': 0.0}
    
    def _detect_shooting_action(self, keypoints_sequence: List[np.ndarray]) -> Dict:
        """Detect shooting action based on leg movement patterns"""
        try:
            # Analyze leg swing motion for shooting
            leg_velocities = []
            for i in range(1, len(keypoints_sequence)):
                if keypoints_sequence[i] is not None and keypoints_sequence[i-1] is not None:
                    # Right leg keypoints (ankle to hip)
                    right_ankle_curr = keypoints_sequence[i][27*4:27*4+2]  # x, y of right ankle
                    right_ankle_prev = keypoints_sequence[i-1][27*4:27*4+2]
                    
                    velocity = np.linalg.norm(right_ankle_curr - right_ankle_prev)
                    leg_velocities.append(velocity)
            
            if leg_velocities:
                max_velocity = max(leg_velocities)
                avg_velocity = np.mean(leg_velocities)
                
                # Simple threshold-based classification
                if max_velocity > 0.05 and avg_velocity > 0.02:
                    return {'action': 'shooting', 'confidence': min(1.0, max_velocity * 10)}
            
            return {'action': 'no_shooting', 'confidence': 0.3}
            
        except Exception as e:
            logger.error(f"Error detecting shooting action: {e}")
            return {'action': 'error', 'confidence': 0.0}
    
    def _detect_passing_action(self, keypoints_sequence: List[np.ndarray]) -> Dict:
        """Detect passing action based on controlled leg movement"""
        try:
            # Similar to shooting but with different velocity thresholds
            leg_movements = self._analyze_leg_movement(keypoints_sequence)
            
            if leg_movements['controlled_movement'] and leg_movements['moderate_speed']:
                return {'action': 'passing', 'confidence': 0.7}
            
            return {'action': 'no_passing', 'confidence': 0.3}
            
        except Exception as e:
            logger.error(f"Error detecting passing action: {e}")
            return {'action': 'error', 'confidence': 0.0}
    
    def _detect_dribbling_action(self, keypoints_sequence: List[np.ndarray]) -> Dict:
        """Detect dribbling based on alternating leg movements"""
        try:
            # Analyze alternating leg pattern
            if len(keypoints_sequence) < 5:
                return {'action': 'insufficient_data', 'confidence': 0.0}
            
            # Check for alternating leg movement pattern
            left_leg_activity = []
            right_leg_activity = []
            
            for i in range(1, len(keypoints_sequence)):
                if keypoints_sequence[i] is not None and keypoints_sequence[i-1] is not None:
                    # Left leg activity
                    left_ankle_movement = np.linalg.norm(
                        keypoints_sequence[i][28*4:28*4+2] - keypoints_sequence[i-1][28*4:28*4+2]
                    )
                    # Right leg activity  
                    right_ankle_movement = np.linalg.norm(
                        keypoints_sequence[i][27*4:27*4+2] - keypoints_sequence[i-1][27*4:27*4+2]
                    )
                    
                    left_leg_activity.append(left_ankle_movement)
                    right_leg_activity.append(right_ankle_movement)
            
            # Check for alternating pattern
            if len(left_leg_activity) > 3:
                alternating_score = self._calculate_alternating_score(left_leg_activity, right_leg_activity)
                if alternating_score > 0.6:
                    return {'action': 'dribbling', 'confidence': alternating_score}
            
            return {'action': 'no_dribbling', 'confidence': 0.3}
            
        except Exception as e:
            logger.error(f"Error detecting dribbling action: {e}")
            return {'action': 'error', 'confidence': 0.0}
    
    def _detect_heading_action(self, keypoints_sequence: List[np.ndarray]) -> Dict:
        """Detect heading action based on head and upper body movement"""
        try:
            head_movements = []
            for i in range(1, len(keypoints_sequence)):
                if keypoints_sequence[i] is not None and keypoints_sequence[i-1] is not None:
                    # Nose keypoint movement (head movement indicator)
                    nose_curr = keypoints_sequence[i][0:2]  # x, y of nose
                    nose_prev = keypoints_sequence[i-1][0:2]
                    
                    head_movement = np.linalg.norm(nose_curr - nose_prev)
                    head_movements.append(head_movement)
            
            if head_movements:
                max_head_movement = max(head_movements)
                if max_head_movement > 0.03:  # Threshold for significant head movement
                    return {'action': 'heading', 'confidence': min(1.0, max_head_movement * 20)}
            
            return {'action': 'no_heading', 'confidence': 0.3}
            
        except Exception as e:
            logger.error(f"Error detecting heading action: {e}")
            return {'action': 'error', 'confidence': 0.0}
    
    def _analyze_leg_movement(self, keypoints_sequence: List[np.ndarray]) -> Dict:
        """Analyze leg movement patterns"""
        movements = []
        for i in range(1, len(keypoints_sequence)):
            if keypoints_sequence[i] is not None and keypoints_sequence[i-1] is not None:
                leg_movement = np.linalg.norm(
                    keypoints_sequence[i][27*4:27*4+2] - keypoints_sequence[i-1][27*4:27*4+2]
                )
                movements.append(leg_movement)
        
        if movements:
            avg_movement = np.mean(movements)
            max_movement = max(movements)
            
            return {
                'controlled_movement': 0.01 < avg_movement < 0.04,
                'moderate_speed': 0.02 < max_movement < 0.06,
                'avg_speed': avg_movement,
                'max_speed': max_movement
            }
        
        return {'controlled_movement': False, 'moderate_speed': False}
    
    def _calculate_alternating_score(self, left_activity: List[float], 
                                   right_activity: List[float]) -> float:
        """Calculate alternating pattern score for dribbling detection"""
        if len(left_activity) != len(right_activity) or len(left_activity) < 3:
            return 0.0
        
        alternating_count = 0
        total_comparisons = 0
        
        for i in range(len(left_activity) - 1):
            total_comparisons += 1
            # Check if activity alternates between legs
            if (left_activity[i] > right_activity[i] and 
                left_activity[i+1] < right_activity[i+1]) or \
               (left_activity[i] < right_activity[i] and 
                left_activity[i+1] > right_activity[i+1]):
                alternating_count += 1
        
        return alternating_count / total_comparisons if total_comparisons > 0 else 0.0

class FootballAnalytics:
    """
    Part 3: Analytics and Correction Suggestion Module
    """
    
    def __init__(self):
        self.technique_standards = {
            'shooting': {
                'body_position': 'balanced',
                'plant_foot': 'stable',
                'striking_leg': 'controlled_swing',
                'follow_through': 'complete'
            },
            'passing': {
                'accuracy': 'high',
                'power_control': 'moderate',
                'body_position': 'stable',
                'foot_contact': 'clean'
            },
            'dribbling': {
                'ball_control': 'close',
                'body_balance': 'low_center',
                'change_of_pace': 'varied',
                'directional_change': 'sharp'
            },
            'heading': {
                'timing': 'precise',
                'contact_point': 'forehead',
                'neck_strength': 'engaged',
                'body_position': 'athletic'
            }
        }
    
    def analyze_technique(self, keypoints_data: List[np.ndarray], 
                         action_type: str) -> Dict:
        """Analyze football technique and provide feedback"""
        
        if not keypoints_data or action_type not in self.technique_standards:
            return {'error': 'Invalid data or action type'}
        
        analysis_result = {
            'action_type': action_type,
            'overall_score': 0.0,
            'detailed_analysis': {},
            'corrections': [],
            'strengths': []
        }
        
        if action_type == 'shooting':
            analysis_result = self._analyze_shooting_technique(keypoints_data)
        elif action_type == 'passing':
            analysis_result = self._analyze_passing_technique(keypoints_data)
        elif action_type == 'dribbling':
            analysis_result = self._analyze_dribbling_technique(keypoints_data)
        elif action_type == 'heading':
            analysis_result = self._analyze_heading_technique(keypoints_data)
        
        return analysis_result
    
    def _analyze_shooting_technique(self, keypoints_data: List[np.ndarray]) -> Dict:
        """Analyze shooting technique and provide specific feedback"""
        analysis = {
            'action_type': 'shooting',
            'overall_score': 0.0,
            'detailed_analysis': {},
            'corrections': [],
            'strengths': []
        }
        
        try:
            # Analyze different aspects of shooting technique
            scores = []
            
            # 1. Body Balance Analysis
            balance_score = self._analyze_body_balance(keypoints_data)
            scores.append(balance_score)
            analysis['detailed_analysis']['body_balance'] = balance_score
            
            if balance_score < 0.6:
                analysis['corrections'].append(
                    "Improve body balance: Keep your head up and shoulders square to the target"
                )
            else:
                analysis['strengths'].append("Good body balance maintained")
            
            # 2. Plant Foot Position
            plant_foot_score = self._analyze_plant_foot_position(keypoints_data)
            scores.append(plant_foot_score)
            analysis['detailed_analysis']['plant_foot'] = plant_foot_score
            
            if plant_foot_score < 0.7:
                analysis['corrections'].append(
                    "Plant foot positioning: Place your non-kicking foot closer to the ball for better accuracy"
                )
            else:
                analysis['strengths'].append("Excellent plant foot positioning")
            
            # 3. Leg Swing Analysis
            leg_swing_score = self._analyze_leg_swing(keypoints_data)
            scores.append(leg_swing_score)
            analysis['detailed_analysis']['leg_swing'] = leg_swing_score
            
            if leg_swing_score < 0.65:
                analysis['corrections'].append(
                    "Leg swing technique: Focus on a smooth, controlled back-swing and accelerate through the ball"
                )
            else:
                analysis['strengths'].append("Good leg swing mechanics")
            
            # 4. Follow Through
            follow_through_score = self._analyze_follow_through(keypoints_data)
            scores.append(follow_through_score)
            analysis['detailed_analysis']['follow_through'] = follow_through_score
            
            if follow_through_score < 0.6:
                analysis['corrections'].append(
                    "Follow through: Complete your kicking motion toward the target"
                )
            else:
                analysis['strengths'].append("Good follow-through execution")
            
            # Calculate overall score
            analysis['overall_score'] = np.mean(scores)
            
            # Overall feedback
            if analysis['overall_score'] >= 0.8:
                analysis['overall_feedback'] = "Excellent shooting technique! Minor refinements can make you even better."
            elif analysis['overall_score'] >= 0.65:
                analysis['overall_feedback'] = "Good shooting form with room for improvement in specific areas."
            else:
                analysis['overall_feedback'] = "Focus on fundamental shooting mechanics. Practice the basics consistently."
            
        except Exception as e:
            logger.error(f"Error analyzing shooting technique: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_passing_technique(self, keypoints_data: List[np.ndarray]) -> Dict:
        """Analyze passing technique"""
        analysis = {
            'action_type': 'passing',
            'overall_score': 0.0,
            'detailed_analysis': {},
            'corrections': [],
            'strengths': []
        }
        
        try:
            scores = []
            
            # Body position analysis
            body_position_score = self._analyze_body_position_passing(keypoints_data)
            scores.append(body_position_score)
            
            if body_position_score < 0.7:
                analysis['corrections'].append(
                    "Body position: Keep your body over the ball and face your target"
                )
            else:
                analysis['strengths'].append("Good body positioning for passing")
            
            # Foot contact analysis
            contact_score = self._analyze_foot_contact(keypoints_data)
            scores.append(contact_score)
            
            if contact_score < 0.6:
                analysis['corrections'].append(
                    "Foot contact: Use the inside of your foot for better accuracy and control"
                )
            else:
                analysis['strengths'].append("Clean foot contact with the ball")
            
            analysis['overall_score'] = np.mean(scores)
            
        except Exception as e:
            logger.error(f"Error analyzing passing technique: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_dribbling_technique(self, keypoints_data: List[np.ndarray]) -> Dict:
        """Analyze dribbling technique"""
        analysis = {
            'action_type': 'dribbling',
            'overall_score': 0.0,
            'detailed_analysis': {},
            'corrections': [],
            'strengths': []
        }
        
        try:
            scores = []
            
            # Center of gravity analysis
            center_gravity_score = self._analyze_center_of_gravity(keypoints_data)
            scores.append(center_gravity_score)
            
            if center_gravity_score < 0.65:
                analysis['corrections'].append(
                    "Lower your center of gravity: Bend your knees more for better ball control"
                )
            else:
                analysis['strengths'].append("Good low center of gravity for dribbling")
            
            # Touch frequency analysis
            touch_frequency_score = self._analyze_touch_frequency(keypoints_data)
            scores.append(touch_frequency_score)
            
            if touch_frequency_score < 0.6:
                analysis['corrections'].append(
                    "Ball touches: Keep the ball closer with more frequent, lighter touches"
                )
            else:
                analysis['strengths'].append("Good ball control with appropriate touches")
            
            analysis['overall_score'] = np.mean(scores)
            
        except Exception as e:
            logger.error(f"Error analyzing dribbling technique: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_heading_technique(self, keypoints_data: List[np.ndarray]) -> Dict:
        """Analyze heading technique"""
        analysis = {
            'action_type': 'heading',
            'overall_score': 0.0,
            'detailed_analysis': {},
            'corrections': [],
            'strengths': []
        }
        
        try:
            scores = []
            
            # Head position analysis
            head_position_score = self._analyze_head_position(keypoints_data)
            scores.append(head_position_score)
            
            if head_position_score < 0.7:
                analysis['corrections'].append(
                    "Head position: Keep your eyes on the ball and use your forehead for contact"
                )
            else:
                analysis['strengths'].append("Good head positioning for headers")
            
            # Body preparation analysis
            body_prep_score = self._analyze_body_preparation_heading(keypoints_data)
            scores.append(body_prep_score)
            
            if body_prep_score < 0.6:
                analysis['corrections'].append(
                    "Body preparation: Arch your back and use your core strength to generate power"
                )
            else:
                analysis['strengths'].append("Good body preparation for heading")
            
            analysis['overall_score'] = np.mean(scores)
            
        except Exception as e:
            logger.error(f"Error analyzing heading technique: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    # Helper methods for specific technique analysis
    def _analyze_body_balance(self, keypoints_data: List[np.ndarray]) -> float:
        """Analyze body balance during movement"""
        try:
            balance_scores = []
            for keypoints in keypoints_data:
                if keypoints is not None:
                    # Calculate shoulder alignment (left vs right shoulder height)
                    left_shoulder = keypoints[11*4:11*4+2]
                    right_shoulder = keypoints[12*4:12*4+2]
                    
                    # Balance is better when shoulders are level
                    height_diff = abs(left_shoulder[1] - right_shoulder[1])
                    balance_score = max(0, 1 - height_diff * 10)  # Scale the difference
                    balance_scores.append(balance_score)
            
            return np.mean(balance_scores) if balance_scores else 0.5
        except:
            return 0.5
    
    def _analyze_plant_foot_position(self, keypoints_data: List[np.ndarray]) -> float:
        """Analyze plant foot positioning"""
        # Simplified analysis - in real implementation, you'd need ball position
        try:
            stability_scores = []
            for keypoints in keypoints_data:
                if keypoints is not None:
                    # Analyze foot stability (less movement = more stable)
                    left_ankle = keypoints[27*4:27*4+2]
                    # Simple stability measure
                    stability_score = 0.75  # Placeholder
                    stability_scores.append(stability_score)
            
            return np.mean(stability_scores) if stability_scores else 0.6
        except:
            return 0.6
    
    def _analyze_leg_swing(self, keypoints_data: List[np.ndarray]) -> float:
        """Analyze leg swing mechanics"""
        try:
            if len(keypoints_data) < 3:
                return 0.5
            
            swing_velocities = []
            for i in range(1, len(keypoints_data)):
                if keypoints_data[i] is not None and keypoints_data[i-1] is not None:
                    ankle_curr = keypoints_data[i][27*4:27*4+2]
                    ankle_prev = keypoints_data[i-1][27*4:27*4+2]
                    velocity = np.linalg.norm(ankle_curr - ankle_prev)
                    swing_velocities.append(velocity)
            
            if swing_velocities:
                # Good swing has moderate, consistent velocity
                avg_velocity = np.mean(swing_velocities)
                velocity_std = np.std(swing_velocities)
                
                # Score based on consistency and appropriate speed
                consistency_score = max(0, 1 - velocity_std * 20)
                speed_score = min(1, avg_velocity * 15)
                
                return (consistency_score + speed_score) / 2
            
            return 0.5
        except:
            return 0.5
    
    def _analyze_follow_through(self, keypoints_data: List[np.ndarray]) -> float:
        """Analyze follow-through motion"""
        try:
            if len(keypoints_data) < 5:
                return 0.5
            
            # Analyze leg position in final frames
            final_frames = keypoints_data[-3:]
            follow_through_quality = 0.7  # Simplified placeholder
            
            return follow_through_quality
        except:
            return 0.5
    
    def _analyze_body_position_passing(self, keypoints_data: List[np.ndarray]) -> float:
        """Analyze body position for passing"""
        return 0.7  # Placeholder implementation
    
    def _analyze_foot_contact(self, keypoints_data: List[np.ndarray]) -> float:
        """Analyze foot contact quality"""
        return 0.65  # Placeholder implementation
    
    def _analyze_center_of_gravity(self, keypoints_data: List[np.ndarray]) -> float:
        """Analyze center of gravity for dribbling"""
        try:
            hip_heights = []
            for keypoints in keypoints_data:
                if keypoints is not None:
                    hip_y = (keypoints[23*4+1] + keypoints[24*4+1]) / 2  # Average hip height
                    hip_heights.append(hip_y)
            
            if hip_heights:
                avg_height = np.mean(hip_heights)
                # Lower height (higher y value in image coordinates) is better for dribbling
                # Assuming normalized coordinates where lower values are higher in image
                low_center_score = min(1, avg_height * 1.5)  # Adjust scaling as needed
                return low_center_score
            
            return 0.6
        except:
            return 0.6
    
    def _analyze_touch_frequency(self, keypoints_data: List[np.ndarray]) -> float:
        """Analyze ball touch frequency"""
        return 0.65  # Placeholder - would need ball tracking
    
    def _analyze_head_position(self, keypoints_data: List[np.ndarray]) -> float:
        """Analyze head position for heading"""
        return 0.7  # Placeholder implementation
    
    def _analyze_body_preparation_heading(self, keypoints_data: List[np.ndarray]) -> float:
        """Analyze body preparation for heading"""
        return 0.65  # Placeholder implementation

class ProjectKittyFootball:
    """
    Main Project KITTY Football System
    Integrates all components for comprehensive football training analysis
    """
    
    def __init__(self, output_dir: str = "project_kitty_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.scraper = FootballVideoScraper(str(self.output_dir / "videos"))
        self.pose_extractor = FootballPoseExtractor()
        self.analytics = FootballAnalytics()
        
        # Results storage
        self.analysis_results = []
        self.player_profiles = {}
    
    def process_video(self, video_path: str, action_type: str = 'shooting') -> Dict:
        """Process a single video and provide complete analysis"""
        logger.info(f"Processing video: {video_path} for {action_type}")
        
        try:
            # Extract keypoints from video
            keypoints_sequence = self._extract_keypoints_from_video(video_path)
            
            if not keypoints_sequence:
                return {'error': 'Failed to extract keypoints from video'}
            
            # Detect football action
            action_detection