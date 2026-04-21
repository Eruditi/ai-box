#!/usr/bin/env python3
"""
教室课堂上课情况分析算法
"""

import cv2
import numpy as np
from typing import Dict, Any, List

from .algorithm_base import AlgorithmBase, AlgorithmResult, AlgorithmCategory


class StudentCountAlgorithm(AlgorithmBase):
    """学生人数统计 - ID: 51"""
    ALGORITHM_ID = 51
    ALGORITHM_NAME = "学生人数统计"
    CATEGORY = AlgorithmCategory.STRUCTURED_ANALYSIS

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_cascade = None

    def initialize(self) -> bool:
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            return True
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.face_cascade is None:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            result.detected = True
            result.confidence = 0.9
            result.extra_data = {
                'student_count': len(faces),
                'faces': [{'x': x, 'y': y, 'w': w, 'h': h} for (x, y, w, h) in faces]
            }

        return result


class StudentAttentionAlgorithm(AlgorithmBase):
    """学生专注度分析 - ID: 52"""
    ALGORITHM_ID = 52
    ALGORITHM_NAME = "学生专注度分析"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_cascade = None
        self.previous_faces = []

    def initialize(self) -> bool:
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            return True
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.face_cascade is None:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            attention_level = 0
            focused_count = 0
            total_count = len(faces)

            for (fx, fy, fw, fh) in faces:
                eye_region = frame[fy + fh//4:fy + fh//2, fx:fx + fw]
                if eye_region.size > 0:
                    eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                    eye_brightness = np.mean(eye_gray)
                    
                    if 80 < eye_brightness < 150:
                        focused_count += 1

            if total_count > 0:
                attention_level = focused_count / total_count

            result.detected = True
            result.confidence = attention_level
            result.extra_data = {
                'total_students': total_count,
                'focused_students': focused_count,
                'attention_level': attention_level
            }

        return result


class ClassroomDisciplineAlgorithm(AlgorithmBase):
    """课堂纪律分析 - ID: 53"""
    ALGORITHM_ID = 53
    ALGORITHM_NAME = "课堂纪律分析"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_cascade = None
        self.previous_frame = None

    def initialize(self) -> bool:
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            return True
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.face_cascade is None:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.previous_frame is None:
            self.previous_frame = gray
            return result

        if self.previous_frame.shape != gray.shape:
            self.previous_frame = gray
            return result

        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        movement_pixels = cv2.countNonZero(thresh)

        discipline_level = 1.0
        if len(faces) > 0:
            movement_ratio = movement_pixels / (frame.shape[0] * frame.shape[1])
            if movement_ratio > 0.1:
                discipline_level = 1.0 - movement_ratio

        result.detected = True
        result.confidence = discipline_level
        result.extra_data = {
            'movement_level': movement_ratio if len(faces) > 0 else 0,
            'discipline_level': discipline_level,
            'student_count': len(faces)
        }

        self.previous_frame = gray
        return result


class TeacherActivityAlgorithm(AlgorithmBase):
    """教师活动分析 - ID: 54"""
    ALGORITHM_ID = 54
    ALGORITHM_NAME = "教师活动分析"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.hog = None
        self.teacher_positions = []

    def initialize(self) -> bool:
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            return True
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.hog is None:
            return result

        if not self._is_safe_for_hog(frame):
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, weights = self.hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

        if len(boxes) > 0:
            largest_person = max(boxes, key=lambda box: box[2] * box[3])
            x, y, w, h = largest_person
            
            self.teacher_positions.append((x, y))
            if len(self.teacher_positions) > 50:
                self.teacher_positions.pop(0)

            activity_level = 0
            if len(self.teacher_positions) > 10:
                positions = np.array(self.teacher_positions)
                movement = np.std(positions, axis=0)
                activity_level = min(1.0, np.mean(movement) / 100)

            result.detected = True
            result.confidence = activity_level
            result.bounding_box = (x, y, w, h)
            result.extra_data = {
                'activity_level': activity_level,
                'teacher_detected': True
            }

        return result


class EquipmentUsageAlgorithm(AlgorithmBase):
    """教学设备使用情况 - ID: 55"""
    ALGORITHM_ID = 55
    ALGORITHM_NAME = "教学设备使用情况"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL

    def initialize(self) -> bool:
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if not self._is_valid_frame(frame):
            return result

        h, w = frame.shape[:2]
        
        screen_region = frame[h//3:h*2//3, w//4:w*3//4]
        if screen_region.size > 0:
            screen_brightness = np.mean(screen_region)
            
            if screen_brightness > 100:
                result.detected = True
                result.confidence = min(1.0, screen_brightness / 255)
                result.bounding_box = (w//4, h//3, w//2, h//3)
                result.extra_data = {
                    'screen_active': True,
                    'brightness': screen_brightness
                }

        return result


class StudentAbsenceAlgorithm(AlgorithmBase):
    """学生缺勤检测 - ID: 56"""
    ALGORITHM_ID = 56
    ALGORITHM_NAME = "学生缺勤检测"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_cascade = None
        self.expected_count = config.get('expected_students', 30)

    def initialize(self) -> bool:
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            return True
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.face_cascade is None:
            return result

        if not self._is_valid_frame(frame):
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        current_count = len(faces)
        absence_count = max(0, self.expected_count - current_count)
        absence_rate = absence_count / self.expected_count if self.expected_count > 0 else 0

        result.detected = True
        result.confidence = absence_rate
        result.extra_data = {
            'current_students': current_count,
            'expected_students': self.expected_count,
            'absent_students': absence_count,
            'absence_rate': absence_rate
        }

        return result


class ClassroomEnvironmentAlgorithm(AlgorithmBase):
    """课堂环境分析 - ID: 57"""
    ALGORITHM_ID = 57
    ALGORITHM_NAME = "课堂环境分析"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL

    def initialize(self) -> bool:
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if not self._is_valid_frame(frame):
            return result

        h, w = frame.shape[:2]
        
        brightness = np.mean(frame)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        clarity = np.var(laplacian)

        environment_score = 0
        if 80 < brightness < 150:
            environment_score += 0.3
        if 50 < saturation < 120:
            environment_score += 0.3
        if clarity > 100:
            environment_score += 0.4

        result.detected = True
        result.confidence = environment_score
        result.extra_data = {
            'brightness': brightness,
            'saturation': saturation,
            'clarity': clarity,
            'environment_score': environment_score
        }

        return result


class StudentInteractionAlgorithm(AlgorithmBase):
    """学生互动分析 - ID: 58"""
    ALGORITHM_ID = 58
    ALGORITHM_NAME = "学生互动分析"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_cascade = None
        self.previous_faces = []

    def initialize(self) -> bool:
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            return True
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.face_cascade is None:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 1:
            # 计算人脸之间的距离，判断互动情况
            interactions = 0
            total_pairs = len(faces) * (len(faces) - 1) // 2
            
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    x1, y1, w1, h1 = faces[i]
                    x2, y2, w2, h2 = faces[j]
                    
                    # 计算中心点距离
                    center1 = (x1 + w1//2, y1 + h1//2)
                    center2 = (x2 + w2//2, y2 + h2//2)
                    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
                    
                    # 如果距离小于两个脸宽之和，认为存在互动
                    if distance < (w1 + w2):
                        interactions += 1
            
            interaction_level = interactions / total_pairs if total_pairs > 0 else 0
            
            result.detected = True
            result.confidence = interaction_level
            result.extra_data = {
                'student_count': len(faces),
                'interaction_count': interactions,
                'interaction_level': interaction_level
            }

        return result


class ClassroomParticipationAlgorithm(AlgorithmBase):
    """课堂参与度分析 - ID: 59"""
    ALGORITHM_ID = 59
    ALGORITHM_NAME = "课堂参与度分析"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_cascade = None
        self.hog = None

    def initialize(self) -> bool:
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            return True
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.face_cascade is None or self.hog is None:
            return result

        if not self._is_safe_for_hog(frame):
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        boxes, _ = self.hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

        if len(faces) > 0:
            # 分析举手或站立的学生
            active_students = 0
            for (x, y, w, h) in boxes:
                # 检测站立的人（高度大于宽度的比例）
                if h > w * 1.5:
                    active_students += 1
            
            participation_rate = active_students / len(faces) if len(faces) > 0 else 0
            
            result.detected = True
            result.confidence = participation_rate
            result.extra_data = {
                'total_students': len(faces),
                'active_students': active_students,
                'participation_rate': participation_rate
            }

        return result


class TeacherTeachingStyleAlgorithm(AlgorithmBase):
    """教师教学方式分析 - ID: 60"""
    ALGORITHM_ID = 60
    ALGORITHM_NAME = "教师教学方式分析"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.hog = None
        self.teacher_positions = []
        self.lecture_mode = 0
        self.interactive_mode = 0

    def initialize(self) -> bool:
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            return True
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.hog is None:
            return result

        if not self._is_safe_for_hog(frame):
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, _ = self.hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

        if len(boxes) > 0:
            largest_person = max(boxes, key=lambda box: box[2] * box[3])
            x, y, w, h = largest_person
            
            self.teacher_positions.append((x, y))
            if len(self.teacher_positions) > 30:
                self.teacher_positions.pop(0)
            
            h_frame, w_frame = frame.shape[:2]
            center_x = w_frame // 2
            
            # 分析教师位置：讲台区域 vs 教室中间
            if x < w_frame // 3:
                self.lecture_mode += 1
            else:
                self.interactive_mode += 1
            
            total_modes = self.lecture_mode + self.interactive_mode
            interactive_ratio = self.interactive_mode / total_modes if total_modes > 0 else 0
            
            teaching_style = "lecture" if interactive_ratio < 0.3 else "interactive" if interactive_ratio > 0.7 else "mixed"
            
            result.detected = True
            result.confidence = interactive_ratio
            result.bounding_box = (x, y, w, h)
            result.extra_data = {
                'teaching_style': teaching_style,
                'interactive_ratio': interactive_ratio,
                'lecture_mode_count': self.lecture_mode,
                'interactive_mode_count': self.interactive_mode
            }

        return result


class ClassroomAtmosphereAlgorithm(AlgorithmBase):
    """课堂氛围分析 - ID: 61"""
    ALGORITHM_ID = 61
    ALGORITHM_NAME = "课堂氛围分析"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_cascade = None
        self.previous_frame = None

    def initialize(self) -> bool:
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            return True
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.face_cascade is None:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.previous_frame is None:
            self.previous_frame = gray
            return result

        if self.previous_frame.shape != gray.shape:
            self.previous_frame = gray
            return result

        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        movement_pixels = cv2.countNonZero(thresh)
        h, w = frame.shape[:2]
        total_pixels = h * w
        movement_ratio = movement_pixels / total_pixels

        # 分析课堂氛围：过于安静 vs 活跃 vs 过于喧闹
        atmosphere_score = 0
        if 0.01 < movement_ratio < 0.05:
            atmosphere_score = 0.8  # 活跃
        elif movement_ratio <= 0.01:
            atmosphere_score = 0.3  # 过于安静
        else:
            atmosphere_score = 0.5  # 过于喧闹

        result.detected = True
        result.confidence = atmosphere_score
        result.extra_data = {
            'movement_ratio': movement_ratio,
            'atmosphere_score': atmosphere_score,
            'student_count': len(faces),
            'atmosphere_level': 'active' if 0.6 < atmosphere_score < 1.0 else 'quiet' if atmosphere_score < 0.4 else 'noisy'
        }

        self.previous_frame = gray
        return result


class StudentEmotionAlgorithm(AlgorithmBase):
    """学生情绪分析 - ID: 62"""
    ALGORITHM_ID = 62
    ALGORITHM_NAME = "学生情绪分析"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_cascade = None

    def initialize(self) -> bool:
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            return True
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.face_cascade is None:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            positive_emotions = 0
            total_faces = len(faces)
            
            for (fx, fy, fw, fh) in faces:
                face_roi = frame[fy:fy+fh, fx:fx+fw]
                if face_roi.size > 0:
                    brightness = np.mean(face_roi)
                    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    contrast = np.std(gray_roi)
                    
                    if 100 < brightness < 180 and 30 < contrast < 80:
                        positive_emotions += 1
            
            positive_ratio = positive_emotions / total_faces if total_faces > 0 else 0
            
            result.detected = True
            result.confidence = positive_ratio
            result.extra_data = {
                'total_students': total_faces,
                'positive_emotion_count': positive_emotions,
                'positive_emotion_ratio': positive_ratio,
                'emotion_level': 'positive' if positive_ratio > 0.6 else 'neutral' if positive_ratio > 0.3 else 'negative'
            }

        return result


class ClassroomRhythmAlgorithm(AlgorithmBase):
    """课堂节奏分析 - ID: 63"""
    ALGORITHM_ID = 63
    ALGORITHM_NAME = "课堂节奏分析"
    CATEGORY = AlgorithmCategory.STRUCTURED_ANALYSIS

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_cascade = None
        self.hog = None
        self.activity_levels = []

    def initialize(self) -> bool:
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            return True
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.face_cascade is None or self.hog is None:
            return result

        if not self._is_safe_for_hog(frame):
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        boxes, _ = self.hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

        current_activity = len(faces) * 0.5 + len(boxes) * 0.5
        self.activity_levels.append(current_activity)
        if len(self.activity_levels) > 20:
            self.activity_levels.pop(0)
        
        if len(self.activity_levels) > 10:
            # 分析节奏变化
            rhythm_change = np.std(self.activity_levels) / np.mean(self.activity_levels) if np.mean(self.activity_levels) > 0 else 0
            
            # 判断节奏是否合适
            rhythm_score = 0
            if 0.1 < rhythm_change < 0.4:
                rhythm_score = 0.8  # 节奏适中
            elif rhythm_change <= 0.1:
                rhythm_score = 0.4  # 节奏过于单调
            else:
                rhythm_score = 0.5  # 节奏过于混乱
            
            result.detected = True
            result.confidence = rhythm_score
            result.extra_data = {
                'current_activity': current_activity,
                'rhythm_change': rhythm_change,
                'rhythm_score': rhythm_score,
                'rhythm_level': 'moderate' if 0.6 < rhythm_score < 1.0 else 'monotonous' if rhythm_score < 0.5 else 'chaotic'
            }

        return result


class TeachingEffectivenessAlgorithm(AlgorithmBase):
    """教学效果评估 - ID: 64"""
    ALGORITHM_ID = 64
    ALGORITHM_NAME = "教学效果评估"
    CATEGORY = AlgorithmCategory.STRUCTURED_ANALYSIS

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_cascade = None
        self.hog = None

    def initialize(self) -> bool:
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            return True
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.face_cascade is None or self.hog is None:
            return result

        if not self._is_safe_for_hog(frame):
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        boxes, _ = self.hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

        if len(faces) > 0:
            effectiveness_score = 0
            
            focused_students = len(faces)
            effectiveness_score += focused_students / len(faces) * 0.3 if len(faces) > 0 else 0
            
            teacher_count = len(boxes)
            if teacher_count > 0:
                effectiveness_score += 0.3
            
            # 3. 课堂互动
            if len(faces) > 1:
                effectiveness_score += 0.2
            
            # 4. 课堂氛围
            h, w = frame.shape[:2]
            brightness = np.mean(frame)
            if 80 < brightness < 150:
                effectiveness_score += 0.2
            
            effectiveness_score = min(1.0, effectiveness_score)
            
            result.detected = True
            result.confidence = effectiveness_score
            result.extra_data = {
                'effectiveness_score': effectiveness_score,
                'student_count': len(faces),
                'teacher_detected': teacher_count > 0,
                'effectiveness_level': 'excellent' if effectiveness_score > 0.8 else 'good' if effectiveness_score > 0.6 else 'average' if effectiveness_score > 0.4 else 'poor'
            }

        return result
