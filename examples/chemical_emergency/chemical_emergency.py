# Copyright (c) 2024.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import os
import logging
import datetime
import json
from typing import Dict, List, Callable, Union, Optional
from dataclasses import dataclass
from graph_of_thoughts import controller, language_models, operations, prompter, parser
from graph_of_thoughts.operations.thought import Thought

logger = logging.getLogger(__name__)

@dataclass
class EmergencyResponse:
    """Data class for emergency response results."""
    situation_analysis: Dict
    impact_assessment: Dict
    response_plan: Dict

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class ChemicalEmergencyPrompter(prompter.Prompter):
    """
    ChemicalEmergencyPrompter provides the generation of prompts specific to the chemical
    emergency response for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    def __init__(self):
        """Initialize the ChemicalEmergencyPrompter."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

    situation_analysis_prompt = """
    你是一名专业的化工安全分析师，请分析以下化工事故情况，并提取关键信息。
    
    事故描述：
    {incident_description}
    
    请分析上述事故，并以JSON格式输出以下信息：
    1. 基本信息（时间、地点、企业名称）
    2. 事故信息（事故类型、事故状态、发展情况）
    3. 气象条件（天气、风向、风速、温度）
    4. 地理信息（区域类型、距离敏感目标的距离）
    5. 敏感目标（周边居民区、学校、医院等）
    
    输出JSON格式如下：
    {{
      "basic_info": {{
        "time": "事发时间（格式：YYYY-MM-DD HH:MM:SS）",
        "location": "事发地点",
        "company": "发生事故的企业名称"
      }},
      "accident_info": {{
        "type": "事故类型（如泄漏、爆炸、火灾等）",
        "status": "事故当前状态（如持续泄漏、已控制等）",
        "development": "事故发展情况（如如何发生的、扩散状况等）"
      }},
      "weather_conditions": {{
        "weather": "天气状况（如晴、雨、雪等）",
        "wind_direction": "风向",
        "wind_speed": "风速",
        "temperature": "温度（摄氏度）"
      }},
      "geographical_info": {{
        "area_type": "区域类型（如工业区、城市、乡村等）",
        "distance_to_residential": "距离最近居民区的距离（米）"
      }},
      "sensitive_targets": {{
        "residential_areas": ["附近居民区列表"],
        "schools": ["附近学校列表"],
        "hospitals": ["附近医院列表"]
      }}
    }}
    
    注意：输出必须严格按照上述JSON格式，所有键名均使用英文。如果某些信息在事故描述中未提及，请根据合理推测填写，或者填写"信息不足"。
    """

    impact_assessment_prompt = """
    你是一名专业的化工安全评估专家，请基于以下化工事故的情景分析，评估潜在影响和后果。
    
    情景分析数据：
    {situation_analysis}
    
    请综合分析上述情景信息，预测事故可能造成的影响和后果，并以JSON格式输出以下评估结果：
    1. 扩散预测（影响半径、主要扩散方向、扩散速度、高中低风险区域）
    2. 人口影响（疏散半径、受影响人口数量、优先疏散区域、预估伤亡）
    3. 环境影响（空气污染、水体污染、土壤污染）
    4. 次生灾害（爆炸风险、火灾风险、有毒释放风险、潜在连锁反应）
    5. 社会影响（经济损失、社会稳定性、行业影响）
    
    输出JSON格式如下：
    {{
      "dispersion_prediction": {{
        "affected_radius": "影响半径（米）",
        "main_direction": "主要扩散方向",
        "spread_speed": "扩散速度估计",
        "high_risk_area": "高风险区域描述",
        "medium_risk_area": "中风险区域描述",
        "low_risk_area": "低风险区域描述"
      }},
      "population_impact": {{
        "evacuation_radius": "建议疏散半径（米）",
        "affected_population": "预估受影响人口数量",
        "priority_evacuation": "优先疏散区域",
        "estimated_casualties": {{
          "severe": "预估重伤人数",
          "moderate": "预估中度伤人数",
          "mild": "预估轻伤人数"
        }}
      }},
      "environmental_impact": {{
        "air_pollution": {{
          "severity": "污染程度（轻微/中度/严重）",
          "duration": "预估持续时间",
          "main_pollutants": "主要污染物"
        }},
        "water_pollution": {{
          "severity": "污染程度（轻微/中度/严重）",
          "affected_water_bodies": "受影响水体",
          "duration": "预估持续时间"
        }},
        "soil_contamination": {{
          "severity": "污染程度（轻微/中度/严重）",
          "affected_area": "受影响面积（平方米）",
          "duration": "预估持续时间"
        }}
      }},
      "secondary_disasters": {{
        "explosion_risk": "爆炸风险评估",
        "fire_risk": "火灾风险评估",
        "toxic_release_risk": "有毒物质释放风险",
        "potential_chain_reactions": "潜在连锁反应"
      }},
      "social_impact": {{
        "economic_loss": "预估经济损失范围",
        "social_stability": "对社会稳定性的影响",
        "industry_impact": "对行业的影响"
      }}
    }}
    
    注意：输出必须严格按照上述JSON格式，所有键名均使用英文。如果某些信息无法基于给定的情景分析进行评估，请标记为"信息不足"。请基于专业知识进行推断，确保评估结果的合理性。
    """

    response_plan_prompt = """
    你是一名专业的化工应急管理专家，请基于以下化工事故的情景分析和影响评估，制定详细的应急响应计划。
    
    事故信息：
    {accident_info}
    
    影响评估：
    {impact_info}
    
    请综合分析上述信息，制定全面的应急响应计划，并以JSON格式输出以下内容：
    1. 应急等级（应急响应级别及确定理由）
    2. 疏散计划（优先疏散区域、疏散路线、集合点、弱势群体安排、交通安排）
    3. 现场应对（隔离区域、泄漏控制方法、危害中和方法、个人防护要求）
    4. 医疗响应（分类救治地点、救护车待命位置、医疗物资、专家团队）
    5. 环境监测（空气质量监测参数及点位、水质监测参数及采样点、报告频率）
    6. 资源配置（应急人员、设备、外部支援）
    7. 信息管理（通知链、公众沟通、谣言控制）
    8. 恢复计划（现场清理、环境修复、生产恢复、长期监测）
    
    输出JSON格式如下：
    {{
      "emergency_level": {{
        "level": "应急响应级别（I级/II级/III级/IV级）",
        "reason": "确定该级别的理由"
      }},
      "evacuation_plan": {{
        "priority_zones": ["优先疏散区域列表"],
        "evacuation_routes": ["推荐疏散路线"],
        "assembly_points": ["集合点位置"],
        "vulnerable_groups_arrangements": "老人、儿童、病患等弱势群体的特殊安排",
        "transportation_arrangements": "交通工具安排"
      }},
      "onsite_response": {{
        "isolation_zone": "建议隔离区域范围",
        "leakage_control": {{
          "method": "控制泄漏的主要方法",
          "backup_plans": ["备选方案"]
        }},
        "hazard_neutralization": {{
          "method": "中和或降低危害的方法",
          "equipment": ["所需设备列表"]
        }},
        "ppe_requirements": "个人防护装备要求"
      }},
      "medical_response": {{
        "triage_locations": ["分类救治地点"],
        "ambulance_standby": "救护车待命位置",
        "medical_supplies": ["所需医疗物资清单"],
        "specialist_team": "所需专家团队组成"
      }},
      "environmental_monitoring": {{
        "air_quality": {{
          "parameters": ["监测参数列表"],
          "monitoring_points": ["监测点位置"]
        }},
        "water_monitoring": {{
          "parameters": ["监测参数列表"],
          "sampling_locations": ["采样点位置"]
        }},
        "reporting_frequency": "监测报告更新频率"
      }},
      "resource_allocation": {{
        "emergency_personnel": {{
          "onsite_command": "现场指挥人员数量和类型",
          "firefighters": "消防人员数量",
          "medical_staff": "医疗人员数量",
          "security": "安保人员数量"
        }},
        "equipment": {{
          "vehicles": ["所需车辆类型及数量"],
          "specialized_equipment": ["所需特殊设备列表"]
        }},
        "external_support": {{
          "government_agencies": ["需联系的政府机构"],
          "nearby_enterprises": ["可能提供支援的周边企业"]
        }}
      }},
      "information_management": {{
        "notification_chain": ["需通知的机构/人员顺序"],
        "public_communication": {{
          "channels": ["信息发布渠道"],
          "frequency": "信息更新频率",
          "content": "关键信息内容要点"
        }},
        "rumor_control": "控制谣言的策略"
      }},
      "recovery_plan": {{
        "site_cleanup": {{
          "methods": "清理方法",
          "timeline": "预计完成时间"
        }},
        "environmental_restoration": {{
          "soil_remediation": "土壤修复方法",
          "vegetation_recovery": "植被恢复计划"
        }},
        "production_resumption": {{
          "safety_inspection": "复产前安全检查要点",
          "equipment_testing": "设备测试要求",
          "staff_training": "员工培训要求"
        }},
        "long_term_monitoring": "长期监测计划"
      }}
    }}
    
    注意：输出必须严格按照上述JSON格式，所有键名均使用英文。如果某些信息无法基于给定的情景分析和影响评估进行规划，请标记为"信息不足"。请基于专业知识进行规划，确保应急响应计划的全面性和可操作性。
    """

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate an aggregation prompt for the language model.

        :param state_dicts: The thought states that should be aggregated.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The aggregation prompt.
        :rtype: str
        """
        # Implement aggregation logic if needed
        return ""

    def generate_prompt(self, state: Dict, operation_index: Optional[int] = None) -> str:
        """
        根据当前状态和操作索引生成提示。

        Args:
            state: 当前状态
            operation_index: 操作索引，用于确定使用哪个提示模板

        Returns:
            str: 生成的提示
        """
        # 使用state中的operation_index，如果没有提供operation_index参数
        if operation_index is None:
            operation_index = state.get("operation_index", 0)

        self.logger.info(f"生成提示，操作索引: {operation_index}")
        
        # 获取输入文本
        input_text = state.get("input_text", "")
        
        # 获取当前状态数据（如果有）
        current_state = state.get("state", {})
        
        # 根据操作索引选择提示模板
        if operation_index == 0:
            # 阶段1：情景分析
            return self.situation_analysis_prompt.format(
                incident_description=input_text
            )
        
        elif operation_index == 1:
            # 阶段2：影响评估
            # 尝试从前一阶段获取结果
            try:
                basic_info = current_state.get("basic_info", {})
                accident_info = current_state.get("accident_info", {})
                weather_conditions = current_state.get("weather_conditions", {})
                geographical_info = current_state.get("geographical_info", {})
                sensitive_targets = current_state.get("sensitive_targets", {})
                
                # 格式化前一阶段数据为JSON字符串用于提示
                situation_analysis_json = json.dumps({
                    "basic_info": basic_info,
                    "accident_info": accident_info,
                    "weather_conditions": weather_conditions, 
                    "geographical_info": geographical_info,
                    "sensitive_targets": sensitive_targets
                }, ensure_ascii=False, indent=2)
                
                return self.impact_assessment_prompt.format(
                    situation_analysis=situation_analysis_json
                )
            except Exception as e:
                self.logger.warning("无法从前一阶段获取状态信息")
                # 提供一个简单的基本提示，避免完全失败
                return self.impact_assessment_prompt.format(
                    situation_analysis="{}"
                )
        
        elif operation_index == 2:
            # 阶段3：响应计划
            # 尝试从前两阶段获取结果
            try:
                # 获取情景分析数据
                basic_info = current_state.get("basic_info", {})
                accident_info = current_state.get("accident_info", {})
                weather_conditions = current_state.get("weather_conditions", {})
                geographical_info = current_state.get("geographical_info", {})
                sensitive_targets = current_state.get("sensitive_targets", {})
                
                # 获取影响评估数据
                impact_assessment = current_state.get("impact_assessment", {})
                
                # 格式化事故信息为JSON字符串
                accident_info_json = json.dumps({
                    "basic_info": basic_info,
                    "accident_info": accident_info,
                    "weather_conditions": weather_conditions,
                    "geographical_info": geographical_info,
                    "sensitive_targets": sensitive_targets
                }, ensure_ascii=False, indent=2)
                
                # 格式化影响评估为JSON字符串
                impact_info_json = json.dumps(impact_assessment, ensure_ascii=False, indent=2)
                
                return self.response_plan_prompt.format(
                    accident_info=accident_info_json,
                    impact_info=impact_info_json
                )
            except Exception as e:
                self.logger.warning("无法从前一阶段获取状态信息")
                # 提供一个简单的基本提示，避免完全失败
            return self.response_plan_prompt.format(
                    accident_info="{}",
                    impact_info="{}"
            )
        
        else:
            # 未知阶段，返回默认提示
            self.logger.warning(f"未知的操作索引: {operation_index}")
            return "请分析这个化工应急事件。"

    def improve_prompt(self, **kwargs) -> str:
        """
        Generate an improve prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The improve prompt.
        :rtype: str
        """
        # Implement improve logic if needed
        return ""

    def validation_prompt(self, **kwargs) -> str:
        """
        Generate a validation prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The validation prompt.
        :rtype: str
        """
        # Implement validation logic if needed
        return ""

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate a score prompt for the language model.

        :param state_dicts: The thought states that should be scored.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The score prompt.
        :rtype: str
        """
        # Implement scoring logic if needed
        return ""

class ChemicalEmergencyParser(parser.Parser):
    """
    ChemicalEmergencyParser provides the parsing of language model responses specific
    to the chemical emergency response example.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self) -> None:
        """Initialize the ChemicalEmergencyParser."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def validate_json_structure(self, json_data: Dict, operation_index: int) -> bool:
        """
        验证JSON结构是否符合指定阶段的要求。

        Args:
            json_data: 要验证的JSON数据
            operation_index: 操作索引，用于确定验证哪个阶段的结构

        Returns:
            bool: 如果结构有效则返回True，否则返回False
        """
        try:
            # 检查json_data是否为空
            if not json_data:
                self.logger.warning(f"JSON数据为空")
                return False
                
            if operation_index == 0:
                # 情景分析阶段 - 宽松验证
                # 检查是否至少包含一些关键字段
                basic_keys = ["basic_info", "accident_info", "weather_conditions", 
                             "geographical_info", "sensitive_targets"]
                
                found_keys = [key for key in basic_keys if key in json_data]
                if len(found_keys) >= 3:  # 至少需要3个主要部分
                    self.logger.info(f"情景分析阶段验证通过，包含字段: {found_keys}")
                    return True
                else:
                    self.logger.warning(f"情景分析缺少太多必要键，仅找到: {found_keys}")
                    return False
                
            elif operation_index == 1:
                # 影响评估阶段 - 宽松验证
                # 检查是否包含关键部分
                impact_keys = ["dispersion_prediction", "population_impact", 
                              "environmental_impact", "secondary_disasters", "social_impact"]
                
                found_keys = [key for key in impact_keys if key in json_data]
                if len(found_keys) >= 2:  # 至少需要2个关键部分
                    self.logger.info(f"影响评估阶段验证通过，包含字段: {found_keys}")
                    return True
                else:
                    self.logger.warning(f"影响评估缺少太多必要键，仅找到: {found_keys}")
                    return False
                
            elif operation_index == 2:
                # 响应计划阶段 - 宽松验证
                # 检查是否包含关键部分
                plan_keys = ["emergency_level", "evacuation_plan", "onsite_response", 
                            "medical_response", "environmental_monitoring", 
                            "resource_allocation", "information_management", "recovery_plan"]
                
                found_keys = [key for key in plan_keys if key in json_data]
                if len(found_keys) >= 3:  # 至少需要3个关键部分
                    self.logger.info(f"响应计划阶段验证通过，包含字段: {found_keys}")
                    return True
                else:
                    self.logger.warning(f"响应计划缺少太多必要键，仅找到: {found_keys}")
                    return False
            
            else:
                # 未知操作索引，默认通过
                self.logger.warning(f"未知操作索引{operation_index}的验证，默认通过")
                return True
                
        except Exception as e:
            self.logger.error(f"验证JSON结构时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """
        Parse the language model response for an aggregation operation.

        :param states: The thought states that were aggregated.
        :type states: List[Dict]
        :param texts: The language model responses to parse.
        :type texts: List[str]
        :return: The parsed response.
        :rtype: Union[Dict, List[Dict]]
        """
        # Implement if aggregation is needed
        return {"state": states[0]["state"]} if states else {}

    def parse_generate_answer(self, thought: Dict, texts: List[str]) -> List[Dict]:
        """
        解析生成操作的响应。

        Args:
            thought: 当前思想状态
            texts: 从语言模型收到的响应文本列表

        Returns:
            List[Dict]: 新的思想状态列表
        """
        new_thoughts = []
        operation_index = thought.get("operation_index", 0)
        input_text = thought.get("input_text", "")

        for text in texts:
            try:
                # 尝试解析响应为JSON
                response_json = {}
                text = text.strip()
                
                # 提取JSON部分
                start_idx = text.find('{')
                end_idx = text.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_text = text[start_idx:end_idx+1]
                    try:
                        response_json = json.loads(json_text)
                        self.logger.info(f"成功解析JSON响应: {json_text[:100]}...")
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON解析错误: {str(e)}")
                        self.logger.error(f"尝试解析的文本: {json_text[:100]}...")
                        continue
                else:
                    self.logger.error(f"无法在文本中找到JSON: {text[:100]}...")
                    continue
                
                # 创建新状态
                new_thought = thought.copy()
                
                # 根据操作索引处理不同阶段的响应
                if operation_index == 0:
                    # 情景分析阶段
                    if self.validate_json_structure(response_json, 0):
                        new_thought["state"] = response_json
                        self.logger.info(f"情景分析阶段：成功验证并保存状态")
                    else:
                        self.logger.warning("情景分析结果未通过验证")
                        continue
                
                elif operation_index == 1:
                    # 影响评估阶段 - 前一阶段结果可能在thought["state"]中
                    if self.validate_json_structure(response_json, 1):
                        # 获取前一阶段的状态
                        prev_state = thought.get("state", {})
                        
                        # 确保我们在新状态中保留前一阶段的数据
                        new_thought["state"] = prev_state.copy()  # 复制前一阶段的整个状态
                        new_thought["state"]["impact_assessment"] = response_json
                        self.logger.info(f"影响评估阶段：成功验证并保存状态")
                    else:
                        self.logger.warning("影响评估结果未通过验证")
                        continue
                
                elif operation_index == 2:
                    # 响应计划阶段
                    if self.validate_json_structure(response_json, 2):
                        # 获取前一阶段的状态
                        prev_state = thought.get("state", {})
                        
                        # 确保我们在新状态中保留前两阶段的数据
                        new_thought["state"] = prev_state.copy()  # 复制前面阶段的整个状态
                        new_thought["state"]["response_plan"] = response_json
                        self.logger.info(f"响应计划阶段：成功验证并保存状态")
                    else:
                        self.logger.warning("响应计划结果未通过验证")
                        continue
                
                # 更新操作索引为下一阶段
                new_thought["operation_index"] = operation_index + 1
                # 确保保留原始输入文本
                new_thought["input_text"] = input_text
                
                # 添加到结果列表
                new_thoughts.append(new_thought)
                
            except Exception as e:
                self.logger.error(f"解析生成响应时出错: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                self.logger.error(f"原始响应: {text[:100]}...")
        
        if not new_thoughts:
            self.logger.warning(f"操作 {operation_index} 没有生成有效的思路")
            # 创建一个基本状态以继续执行
            basic_thought = thought.copy()
            basic_thought["operation_index"] = operation_index + 1
            # 确保保留原始输入文本
            basic_thought["input_text"] = input_text
            if "state" not in basic_thought:
                basic_thought["state"] = {}
            new_thoughts.append(basic_thought)
        
        return new_thoughts

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        Parse the language model response for an improve operation.

        :param state: The current thought state.
        :type state: Dict
        :param texts: The language model responses to parse.
        :type texts: List[str]
        :return: The parsed response.
        :rtype: Dict
        """
        # Implement if improve operation is needed
        return state

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """
        Parse the language model response for a validation operation.

        :param state: The current thought state.
        :type state: Dict
        :param texts: The language model responses to parse.
        :type texts: List[str]
        :return: Whether the state is valid.
        :rtype: bool
        """
        # Implement if validation is needed
        return True

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """
        Parse the language model response for a score operation.

        :param states: The thought states that were scored.
        :type states: List[Dict]
        :param texts: The language model responses to parse.
        :type texts: List[str]
        :return: The parsed scores.
        :rtype: List[float]
        """
        # Implement if scoring is needed
        return [1.0] * len(states)

def validate_input(input_text: str) -> bool:
    """
    验证输入文本是否包含所需信息。

    :param input_text: 要验证的输入文本。
    :type input_text: str
    :return: 如果有效返回True，否则返回False。
    :rtype: bool
    """
    required_info = {
        "时间": ["年", "月", "日", "点"],
        "位置": ["省", "市", "区", "厂"],
        "事故": ["事故", "泄漏", "泄露"],
        "天气": ["天气", "晴", "阴", "雨"],
        "温度": ["温度", "℃"],
        "风": ["风"],
        "距离": ["距离", "公里", "米"],
        "化学品": ["化学品", "氯气"]
    }
    
    missing_info = []
    for key, keywords in required_info.items():
        if not any(keyword in input_text for keyword in keywords):
            missing_info.append(key)
    
    if missing_info:
        logging.getLogger(__name__).warning(f"缺少必要信息: {', '.join(missing_info)}")
        return False
    
    return True

def create_operations_graph(input_text: str) -> controller.Controller:
    """
    创建化工应急响应的操作图。

    Args:
        input_text: 输入文本
    
    Returns:
        Controller: 控制器实例
    """
    # 创建自定义生成操作
    class IndexedGenerate(operations.Generate):
        def __init__(self, operation_index=0, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.operation_index = operation_index
            self.logger = logging.getLogger(__name__)
        
        def _execute(self, lm, prompter, parser, **kwargs):
            """
            执行操作，将operation_index添加到状态中。
            
            Args:
                lm: 语言模型
                prompter: 提示生成器
                parser: 解析器
                **kwargs: 其他参数
            """
            # 确保思想状态中有操作索引
            kwargs["operation_index"] = self.operation_index
            # 确保思想状态中有输入文本
            if "input_text" not in kwargs:
                kwargs["input_text"] = input_text
            
            self.logger.info(f"执行操作类型 {self.__class__.__name__}，操作索引: {self.operation_index}")
            
            # 获取前序思想
            previous_thoughts = self.get_previous_thoughts()
            
            if len(previous_thoughts) == 0 and len(self.predecessors) > 0:
                return
                
            if len(previous_thoughts) == 0:
                # 没有前序操作，使用kwargs作为基础状态
                previous_thoughts = [operations.Thought(state=kwargs)]
                
            for thought in previous_thoughts:
                # 获取基础状态
                base_state = thought.state
                # 调用prompter生成提示
                prompt = prompter.generate_prompt(state=base_state, operation_index=self.operation_index)
                self.logger.debug(f"为LM生成的提示: {prompt}")
                
                # 查询语言模型
                responses = lm.get_response_texts(
                    lm.query(prompt, num_responses=self.num_branches_response)
                )
                self.logger.debug(f"LM的响应: {responses}")
                
                # 解析响应
                new_states = parser.parse_generate_answer(thought.state, responses)
                
                # 创建新思想
                for new_state in new_states:
                    self.thoughts.append(operations.Thought(new_state))
    
    # 创建 LLM 后端
    config_path = os.path.join(
        os.path.dirname(__file__),
        "../../graph_of_thoughts/language_models/config.json",
    )
    lm = language_models.ChatGPT(
        config_path,
        model_name="chatgpt",
        cache=True,
    )
    
    # 创建提示生成器和解析器
    prompter = ChemicalEmergencyPrompter()
    parser = ChemicalEmergencyParser()  # 使用正确的解析器类
    
    # 创建操作图
    graph = operations.GraphOfOperations()
    
    # 阶段1：情景分析
    situation_analysis_op = IndexedGenerate(
        operation_index=0,
            num_branches_prompt=1,
            num_branches_response=1
    )
    
    # 阶段2：影响评估
    impact_assessment_op = IndexedGenerate(
        operation_index=1,
            num_branches_prompt=1,
            num_branches_response=1
    )
    
    # 阶段3：响应计划
    response_plan_op = IndexedGenerate(
        operation_index=2,
            num_branches_prompt=1,
            num_branches_response=1
        )
    
    # 添加操作到图
    graph.append_operation(situation_analysis_op)
    graph.append_operation(impact_assessment_op)
    graph.append_operation(response_plan_op)
    
    # 设置操作之间的依赖关系
    situation_analysis_op.add_successor(impact_assessment_op)
    impact_assessment_op.add_successor(response_plan_op)
    
    # 创建控制器
    controller_instance = controller.Controller(
        lm,
        graph,
        prompter,
        parser,
        {
            "input_text": input_text,
            "operation_index": 0,
            "state": {}
        }
    )
    
    return controller_instance

def process_results(result: List[Dict]) -> EmergencyResponse:
    """
    Process and validate the results from the GOT system.

    :param result: The raw results from the GOT system.
    :type result: List[Dict]
    :return: Processed emergency response results.
    :rtype: EmergencyResponse
    """
    if len(result) != 3:
        raise ValueError("Expected results from all three phases")

    return EmergencyResponse(
        situation_analysis=result[0]["state"],
        impact_assessment=result[1]["state"],
        response_plan=result[2]["state"]
    )

def save_results(results, output_file="emergency_response_plan.json"):
    """将结果保存到文件。"""
    if not results:
        logging.warning("没有结果可保存")
        return
    
    logger = logging.getLogger(__name__)
    
    # 创建存储每个阶段结果的字典
    response = {}
    phase_names = ["situation_analysis", "impact_assessment", "response_plan"]
    
    # 收集已完成阶段的结果
    completed_phases = 0
    
    # 检查获得了多少个阶段的结果
    logger.info(f"获得了 {len(results)} 个阶段的结果")
    
    # 处理每个阶段的结果
    for i, phase_thoughts in enumerate(results):
        if i >= len(phase_names):
            break
            
        if phase_thoughts and len(phase_thoughts) > 0:
            try:
                # 获取该阶段最后一个思路的状态
                last_thought = phase_thoughts[-1]
                logger.info(f"处理阶段 {i} ({phase_names[i]}) 的结果")
                
                if "state" in last_thought.state:
                    # 直接使用状态
                    response[phase_names[i]] = last_thought.state["state"]
                    logger.info(f"阶段 {i} 的状态包含在 'state' 字段中")
                else:
                    # 将整个状态作为结果
                    response[phase_names[i]] = last_thought.state
                    logger.info(f"阶段 {i} 使用完整状态作为结果")
                
                # 标记阶段完成
                completed_phases += 1
                logger.info(f"成功提取阶段 {i} ({phase_names[i]}) 的结果")
            except Exception as e:
                logger.error(f"提取阶段 {i} 结果时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                response[phase_names[i]] = {"error": f"无法提取结果: {str(e)}"}
    
    # 保存结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        
        if completed_phases == len(phase_names):
            logger.info(f"完整的应急响应方案已保存到 {output_file}")
        else:
            logger.warning(f"不完整的应急响应已保存到 {output_file}。只完成了 {completed_phases}/{len(phase_names)} 个阶段。")
    except Exception as e:
        logger.error(f"保存结果到文件时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def setup_logging() -> logging.Logger:
    """
    Set up logging configuration.

    :return: Configured logger instance.
    :rtype: logging.Logger
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("chemical_emergency.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """
    运行化工应急响应系统的主函数。
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # 示例输入文本
        input_text = """
        2024年3月15日上午10点，位于江苏省南京市江北新区的某化工企业发生化学品泄漏事故。
        事故发生在一个储存有氯气的储罐，由于阀门故障导致泄漏。
        事发时天气晴朗，温度20℃，东南风3级。
        泄漏点位于厂区东北角，距离最近的居民区约2公里。
        """

        # 创建控制器
        got_controller = create_operations_graph(input_text)

        # 执行操作
        logger.info("开始执行化工应急响应分析...")
        got_controller.run()
        logger.info("应急响应分析完成")

        # 获取所有阶段结果
        results = got_controller.get_final_thoughts()
        
        # 创建存储各阶段结果的字典
        response = {}
        
        # 检查是否至少有一个阶段的结果
        if results and len(results) > 0 and len(results[0]) > 0:
            # 获取最后一个思路的状态，它应该包含所有阶段的结果
            final_thought = results[0][-1]
            
            # 检查状态是否存在
            if "state" in final_thought.state:
                state = final_thought.state["state"]
                logger.info(f"从最终状态中提取结果")
                
                # 提取各个阶段的结果
                if "basic_info" in state and "accident_info" in state:
                    # 情景分析阶段
                    situation_analysis = {
                        "basic_info": state.get("basic_info", {}),
                        "accident_info": state.get("accident_info", {}),
                        "weather_conditions": state.get("weather_conditions", {}),
                        "geographical_info": state.get("geographical_info", {}),
                        "sensitive_targets": state.get("sensitive_targets", {})
                    }
                    response["situation_analysis"] = situation_analysis
                    logger.info(f"成功提取情景分析阶段的结果")
                
                if "impact_assessment" in state:
                    # 影响评估阶段
                    response["impact_assessment"] = state["impact_assessment"]
                    logger.info(f"成功提取影响评估阶段的结果")
                
                if "response_plan" in state:
                    # 响应计划阶段
                    response["response_plan"] = state["response_plan"]
                    logger.info(f"成功提取响应计划阶段的结果")
            else:
                logger.warning("无法从最终状态中提取结果")
        else:
            logger.warning("未能获取任何阶段的结果")
        
        # 保存结果
        output_file = "emergency_response_plan.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
            
            completed_phases = len(response)
            if completed_phases == 3:
                logger.info(f"完整的应急响应方案已保存到 {output_file}")
            else:
                logger.warning(f"不完整的应急响应已保存到 {output_file}。只完成了 {completed_phases}/3 个阶段。")
        except Exception as e:
            logger.error(f"保存结果到文件时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
        # 创建一个包含所有三个阶段结果的调试文件
        debug_file = "debug_all_results.json"
        all_results = {}
        
        for i, phase_thoughts in enumerate(results):
            if phase_thoughts and len(phase_thoughts) > 0:
                phase_key = f"phase_{i}"
                # 创建可序列化的状态副本
                state_copy = dict(phase_thoughts[-1].state)
                if "graph" in state_copy:
                    del state_copy["graph"]
                all_results[phase_key] = state_copy
        
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"所有阶段结果已保存到 {debug_file}")

    except Exception as e:
        logger.error(f"发生意外错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 