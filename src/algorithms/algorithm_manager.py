#!/usr/bin/env python3
"""
算法管理器 - 统一管理所有50种算法
"""

import time
import logging
from typing import Dict, List, Any, Optional
import cv2
import numpy as np

from .algorithm_base import AlgorithmBase, AlgorithmResult, AlgorithmCategory


class AlgorithmManager:
    """算法管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.algorithms: Dict[int, AlgorithmBase] = {}
        self.algorithm_configs: Dict[int, Dict[str, Any]] = {}
        self._load_algorithm_configs()
        self._register_all_algorithms()

    def _load_algorithm_configs(self):
        """加载算法配置"""
        self.algorithm_configs = {
            1: {'name': '未佩戴安全帽报警', 'category': AlgorithmCategory.PERSON_VIOLATION, 'enabled': True},
            2: {'name': '未戴口罩报警', 'category': AlgorithmCategory.PERSON_VIOLATION, 'enabled': True},
            3: {'name': '未穿戴工作服报警', 'category': AlgorithmCategory.PERSON_VIOLATION, 'enabled': True},
            4: {'name': '未佩戴安全带报警', 'category': AlgorithmCategory.PERSON_VIOLATION, 'enabled': True},
            5: {'name': '未佩戴反光衣报警', 'category': AlgorithmCategory.PERSON_VIOLATION, 'enabled': True},
            6: {'name': '火焰报警', 'category': AlgorithmCategory.ENVIRONMENT_ABNORMAL, 'enabled': True},
            7: {'name': '烟雾报警', 'category': AlgorithmCategory.ENVIRONMENT_ABNORMAL, 'enabled': True},
            8: {'name': '消防设施检测', 'category': AlgorithmCategory.ENVIRONMENT_ABNORMAL, 'enabled': True},
            9: {'name': '杂物堆放', 'category': AlgorithmCategory.ENVIRONMENT_ABNORMAL, 'enabled': True},
            10: {'name': '车辆禁停', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            11: {'name': '车辆离开', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            12: {'name': '人员徘徊', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            13: {'name': '翻墙检测', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            14: {'name': '入侵', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            15: {'name': '越界', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            16: {'name': '摔倒检测', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            17: {'name': '抽烟检测', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            18: {'name': '打电话', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            19: {'name': '看手机', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            20: {'name': '人员奔跑', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            21: {'name': '睡岗检测', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            22: {'name': '人员离岗', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            23: {'name': '人员聚众', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            24: {'name': '人员扭打', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            25: {'name': '人脸', 'category': AlgorithmCategory.STRUCTURED_ANALYSIS, 'enabled': True},
            26: {'name': '人形', 'category': AlgorithmCategory.STRUCTURED_ANALYSIS, 'enabled': True},
            27: {'name': '机动车', 'category': AlgorithmCategory.STRUCTURED_ANALYSIS, 'enabled': True},
            28: {'name': '非机动车', 'category': AlgorithmCategory.STRUCTURED_ANALYSIS, 'enabled': True},
            29: {'name': '车牌', 'category': AlgorithmCategory.STRUCTURED_ANALYSIS, 'enabled': True},
            30: {'name': '人脸识别对比', 'category': AlgorithmCategory.FACE_RECOGNITION, 'enabled': True},
            31: {'name': '车牌识别对比', 'category': AlgorithmCategory.FACE_RECOGNITION, 'enabled': True},
            32: {'name': '超员', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            33: {'name': '少员', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            34: {'name': '人员离开', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            35: {'name': '非机动车禁停', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            36: {'name': '非机动车离开', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            37: {'name': '骑车未带安全帽', 'category': AlgorithmCategory.PERSON_VIOLATION, 'enabled': True},
            38: {'name': '机动车超出数量', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            39: {'name': '机动车少于数量', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            40: {'name': '危化品车辆禁入', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            41: {'name': '骑摩托车进加油站', 'category': AlgorithmCategory.PERSON_VIOLATION, 'enabled': True},
            42: {'name': '卸油流程不规范', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            43: {'name': '标识牌识别', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            44: {'name': '人员滞留', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            45: {'name': '举手求救', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            46: {'name': '人脸识别报警', 'category': AlgorithmCategory.FACE_RECOGNITION, 'enabled': True},
            47: {'name': '摄像头遮挡', 'category': AlgorithmCategory.ENVIRONMENT_ABNORMAL, 'enabled': True},
            48: {'name': '摄像头偏移', 'category': AlgorithmCategory.ENVIRONMENT_ABNORMAL, 'enabled': True},
            49: {'name': '跑冒滴漏', 'category': AlgorithmCategory.ENVIRONMENT_ABNORMAL, 'enabled': True},
            50: {'name': '疲劳驾驶', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            51: {'name': '学生人数统计', 'category': AlgorithmCategory.STRUCTURED_ANALYSIS, 'enabled': True},
            52: {'name': '学生专注度分析', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            53: {'name': '课堂纪律分析', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            54: {'name': '教师活动分析', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            55: {'name': '教学设备使用情况', 'category': AlgorithmCategory.ENVIRONMENT_ABNORMAL, 'enabled': True},
            56: {'name': '学生缺勤检测', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            57: {'name': '课堂环境分析', 'category': AlgorithmCategory.ENVIRONMENT_ABNORMAL, 'enabled': True},
            58: {'name': '学生互动分析', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            59: {'name': '课堂参与度分析', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            60: {'name': '教师教学方式分析', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            61: {'name': '课堂氛围分析', 'category': AlgorithmCategory.ENVIRONMENT_ABNORMAL, 'enabled': True},
            62: {'name': '学生情绪分析', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            63: {'name': '课堂节奏分析', 'category': AlgorithmCategory.STRUCTURED_ANALYSIS, 'enabled': True},
            64: {'name': '教学效果评估', 'category': AlgorithmCategory.STRUCTURED_ANALYSIS, 'enabled': True},
            65: {'name': '牲畜检测', 'category': AlgorithmCategory.STRUCTURED_ANALYSIS, 'enabled': True},
            66: {'name': '禁牧区域识别', 'category': AlgorithmCategory.PERIMETER_ALERT, 'enabled': True},
            67: {'name': '放牧监控', 'category': AlgorithmCategory.BEHAVIOR_ALERT, 'enabled': True},
            68: {'name': '无人机火焰检测', 'category': AlgorithmCategory.ENVIRONMENT_ABNORMAL, 'enabled': True},
            69: {'name': '无人机烟火综合分析', 'category': AlgorithmCategory.ENVIRONMENT_ABNORMAL, 'enabled': True},
        }

    def _register_all_algorithms(self):
        """注册所有算法"""
        from .person_violation import (
            NoHelmetAlgorithm, NoMaskAlgorithm, NoWorkwearAlgorithm,
            NoSafetyBeltAlgorithm, NoReflectiveVestAlgorithm, NoHelmetRidingAlgorithm,
            MotorcycleInGasStationAlgorithm
        )
        from .environment_abnormal import (
            FireDetectionAlgorithm, SmokeDetectionAlgorithm, FireEquipmentAlgorithm,
            DebrisDetectionAlgorithm, CameraOcclusionAlgorithm, CameraShiftAlgorithm,
            LeakDetectionAlgorithm
        )
        from .perimeter_alert import (
            VehicleParkingAlgorithm, VehicleLeaveAlgorithm, PersonLoiteringAlgorithm,
            ClimbOverAlgorithm, IntrusionAlgorithm, CrossBorderAlgorithm,
            OverCapacityAlgorithm, UnderCapacityAlgorithm, PersonLeaveAlgorithm,
            NonMotorParkingAlgorithm, NonMotorLeaveAlgorithm, VehicleOverCountAlgorithm,
            VehicleUnderCountAlgorithm, HazardVehicleAlgorithm, UnloadingProcedureAlgorithm,
            SignDetectionAlgorithm
        )
        from .behavior_alert import (
            FallDetectionAlgorithm, SmokingDetectionAlgorithm, PhoneCallAlgorithm,
            PhoneUsingAlgorithm, RunningAlgorithm, SleepingOnJobAlgorithm,
            PersonAbsentAlgorithm, CrowdGatheringAlgorithm, FightingAlgorithm,
            PersonLoiteringBehaviorAlgorithm, HelpGestureAlgorithm, FatigueDrivingAlgorithm
        )
        from .structured_analysis import (
            FaceDetectionAlgorithm, HumanShapeAlgorithm, MotorVehicleAlgorithm,
            NonMotorVehicleAlgorithm, LicensePlateAlgorithm
        )
        from .face_recognition import (
            FaceRecognitionCompareAlgorithm, LicensePlateCompareAlgorithm,
            FaceRecognitionAlertAlgorithm
        )
        from .classroom_analysis import (
            StudentCountAlgorithm, StudentAttentionAlgorithm, ClassroomDisciplineAlgorithm,
            TeacherActivityAlgorithm, EquipmentUsageAlgorithm, StudentAbsenceAlgorithm,
            ClassroomEnvironmentAlgorithm, StudentInteractionAlgorithm, ClassroomParticipationAlgorithm,
            TeacherTeachingStyleAlgorithm, ClassroomAtmosphereAlgorithm, StudentEmotionAlgorithm,
            ClassroomRhythmAlgorithm, TeachingEffectivenessAlgorithm
        )
        from .grazing_prohibition import (
            LivestockDetectionAlgorithm, GrazingProhibitionAlgorithm, GrazingMonitoringAlgorithm
        )
        from .drone_fire_detection import (
            DroneFireDetectionAlgorithm, DroneFireSmokeAnalyzer
        )

        algorithm_classes = [
            (1, NoHelmetAlgorithm),
            (2, NoMaskAlgorithm),
            (3, NoWorkwearAlgorithm),
            (4, NoSafetyBeltAlgorithm),
            (5, NoReflectiveVestAlgorithm),
            (6, FireDetectionAlgorithm),
            (7, SmokeDetectionAlgorithm),
            (8, FireEquipmentAlgorithm),
            (9, DebrisDetectionAlgorithm),
            (10, VehicleParkingAlgorithm),
            (11, VehicleLeaveAlgorithm),
            (12, PersonLoiteringAlgorithm),
            (13, ClimbOverAlgorithm),
            (14, IntrusionAlgorithm),
            (15, CrossBorderAlgorithm),
            (16, FallDetectionAlgorithm),
            (17, SmokingDetectionAlgorithm),
            (18, PhoneCallAlgorithm),
            (19, PhoneUsingAlgorithm),
            (20, RunningAlgorithm),
            (21, SleepingOnJobAlgorithm),
            (22, PersonAbsentAlgorithm),
            (23, CrowdGatheringAlgorithm),
            (24, FightingAlgorithm),
            (25, FaceDetectionAlgorithm),
            (26, HumanShapeAlgorithm),
            (27, MotorVehicleAlgorithm),
            (28, NonMotorVehicleAlgorithm),
            (29, LicensePlateAlgorithm),
            (30, FaceRecognitionCompareAlgorithm),
            (31, LicensePlateCompareAlgorithm),
            (32, OverCapacityAlgorithm),
            (33, UnderCapacityAlgorithm),
            (34, PersonLeaveAlgorithm),
            (35, NonMotorParkingAlgorithm),
            (36, NonMotorLeaveAlgorithm),
            (37, NoHelmetRidingAlgorithm),
            (38, VehicleOverCountAlgorithm),
            (39, VehicleUnderCountAlgorithm),
            (40, HazardVehicleAlgorithm),
            (41, MotorcycleInGasStationAlgorithm),
            (42, UnloadingProcedureAlgorithm),
            (43, SignDetectionAlgorithm),
            (44, PersonLoiteringBehaviorAlgorithm),
            (45, HelpGestureAlgorithm),
            (46, FaceRecognitionAlertAlgorithm),
            (47, CameraOcclusionAlgorithm),
            (48, CameraShiftAlgorithm),
            (49, LeakDetectionAlgorithm),
            (50, FatigueDrivingAlgorithm),
            (51, StudentCountAlgorithm),
            (52, StudentAttentionAlgorithm),
            (53, ClassroomDisciplineAlgorithm),
            (54, TeacherActivityAlgorithm),
            (55, EquipmentUsageAlgorithm),
            (56, StudentAbsenceAlgorithm),
            (57, ClassroomEnvironmentAlgorithm),
            (58, StudentInteractionAlgorithm),
            (59, ClassroomParticipationAlgorithm),
            (60, TeacherTeachingStyleAlgorithm),
            (61, ClassroomAtmosphereAlgorithm),
            (62, StudentEmotionAlgorithm),
            (63, ClassroomRhythmAlgorithm),
            (64, TeachingEffectivenessAlgorithm),
            (65, LivestockDetectionAlgorithm),
            (66, GrazingProhibitionAlgorithm),
            (67, GrazingMonitoringAlgorithm),
            (68, DroneFireDetectionAlgorithm),
            (69, DroneFireSmokeAnalyzer),
        ]

        for algo_id, algo_class in algorithm_classes:
            if algo_id in self.algorithm_configs:
                config = self.algorithm_configs[algo_id]
                try:
                    self.algorithms[algo_id] = algo_class(config)
                    logging.info(f"Registered algorithm: {algo_id} - {config['name']}")
                except Exception as e:
                    logging.error(f"Failed to register algorithm {algo_id}: {e}")

    def initialize_all(self) -> bool:
        """初始化所有算法"""
        success = True
        for algo_id, algorithm in self.algorithms.items():
            try:
                if algorithm.initialize():
                    logging.info(f"Initialized algorithm: {algo_id} - {algorithm.ALGORITHM_NAME}")
                else:
                    logging.warning(f"Failed to initialize algorithm: {algo_id}")
                    success = False
            except Exception as e:
                logging.error(f"Error initializing algorithm {algo_id}: {e}")
                success = False
        return success

    def process_frame(self, frame: np.ndarray, enabled_algorithms: List[int] = None, 
                     context: Dict[str, Any] = None) -> List[AlgorithmResult]:
        """处理帧并返回所有启用算法的结果"""
        results = []
        context = context or {}
        
        algorithms_to_process = enabled_algorithms or self.algorithms.keys()
        
        for algo_id in algorithms_to_process:
            if algo_id not in self.algorithms:
                continue
                
            algorithm = self.algorithms[algo_id]
            if not algorithm.enabled:
                continue
                
            try:
                result = algorithm.process(frame, context)
                result.timestamp = time.time()
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing algorithm {algo_id}: {e}")
        
        return results

    def visualize_results(self, frame: np.ndarray, results: List[AlgorithmResult]) -> np.ndarray:
        """可视化所有算法结果"""
        vis_frame = frame.copy()
        for result in results:
            if result.detected and result.algorithm_id in self.algorithms:
                algorithm = self.algorithms[result.algorithm_id]
                vis_frame = algorithm.visualize(vis_frame, result)
        return vis_frame

    def get_algorithm(self, algo_id: int) -> Optional[AlgorithmBase]:
        """获取指定算法"""
        return self.algorithms.get(algo_id)

    def enable_algorithm(self, algo_id: int, enabled: bool = True):
        """启用/禁用算法"""
        if algo_id in self.algorithms:
            self.algorithms[algo_id].enabled = enabled

    def get_all_algorithm_info(self) -> List[Dict[str, Any]]:
        """获取所有算法信息"""
        info_list = []
        for algo_id, config in self.algorithm_configs.items():
            algorithm = self.algorithms.get(algo_id)
            info_list.append({
                'id': algo_id,
                'name': config['name'],
                'category': config['category'].name,
                'enabled': algorithm.enabled if algorithm else False
            })
        return info_list

    def release(self):
        """释放所有算法资源"""
        for algorithm in self.algorithms.values():
            try:
                algorithm.release()
            except Exception as e:
                logging.error(f"Error releasing algorithm: {e}")
