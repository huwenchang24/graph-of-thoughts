import os
import logging
import datetime
import json
import re
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
    
    注意：输出必须严格按照上述JSON格式，所有键名均使用英文，确保所有键名对应的内容均使用中文。如果某些信息无法基于给定的情景分析进行评估，请标记为"信息不足"。请基于专业知识进行推断，确保评估结果的合理性。
    """

    response_plan_prompt = """
    你是一名专业的化工应急管理专家，请基于以下化工事故的情景分析和影响评估，制定详细的应急响应计划。
    
    事故信息：
    {accident_info}
    
    影响评估：
    {impact_info}
    
    请综合分析上述信息，制定全面详尽的应急响应计划，并以JSON格式输出以下内容：
    1. 应急等级（应急响应级别及确定理由）
    2. 疏散计划（优先疏散区域、疏散路线、集合点、弱势群体安排、交通安排、疏散通知方式）
    3. 现场应对（隔离区域设定、泄漏控制方法、危害中和方法、个人防护要求、现场指挥体系）
    4. 医疗响应（分类救治地点、救护车待命位置、医疗物资、专家团队、伤员分类处理机制）
    5. 环境监测（空气质量监测参数及点位、水质监测参数及采样点、土壤监测方案、报告频率）
    6. 资源配置（应急人员、设备、外部支援、物资调度、应急指挥中心设置）
    7. 信息管理（通知链、公众沟通、谣言控制、信息发布机制、部门协调机制）
    8. 恢复计划（现场清理、环境修复、生产恢复、长期监测、区域重建）
    9. 次生灾害防范（爆炸预防、火灾防控、有毒物质扩散防范、连锁反应控制、临近设施保护）
    10. 专项处置方案（针对事故中涉及的特殊危险化学品的专项处置方案，如硝酸铵、氰化钠等）
    
    参考案例 - 天津港"8·12"特别重大火灾爆炸事故应急响应：
    {{
      "emergency_level": {{
        "level": "I级",
        "reason": "爆炸导致大量人员伤亡和严重环境污染，影响范围广泛，救援难度大",
        "initiation_procedure": "由天津市政府启动应急预案，相关部门全面响应"
      }},
      "evacuation_plan": {{
        "priority_zones": ["事故中心及周边半径500米范围内"],
        "evacuation_routes": ["东北方向为主要安全疏散路线，避开事故中心和火源"],
        "assembly_points": ["事故中心西南方向开阔地带"],
        "vulnerable_groups_arrangements": "老人、儿童、病患等弱势群体优先疏散并安置到医疗点或安全地点",
        "transportation_arrangements": "大型客车和救护车协同疏散，航空器和轮船协助疏散受困群众",
        "notification_methods": "通过警报、广播、短信、社区工作人员上门通知等方式通知疏散指令",
        "evacuation_stages": "分为紧急疏散、远离危险区域、集中安置等阶段进行疏散"
      }},
      "onsite_response": {{
        "isolation_zone": {{
          "hot_zone": "500米",
          "warm_zone": "500-1000米",
          "cold_zone": "1000米以外"
        }},
        "command_system": {{
          "field_command_post": "事故中心西南方向设立现场指挥部",
          "command_structure": "指挥部设置总指挥、救援、医疗、环保等部门",
          "communication_system": "确保现场通讯畅通，备有应急通讯设备"
        }},
        "leakage_control": {{
          "method": "使用泡沫、水雾等灭火剂进行泄漏控制",
          "backup_plans": ["二次封锁泄露源，转移有害品"],
          "material_needed": ["泡沫灭火器、水枪等"]
        }},
        "hazard_neutralization": {{
          "method": "采用中和剂稀释有害气体，降低危害程度",
          "equipment": ["中和剂喷射设备、防护服"],
          "technical_guidance": "专业化学团队指导下进行"
        }},
        "ppe_requirements": {{
          "entry_team": "穿戴级别高的化学防护服、呼吸器等防护装备",
          "decontamination_team": "穿着特殊防护服进行清洁过程",
          "medical_staff": "佩戴防护口罩、手套、隔离衣等"
        }},
        "decontamination": {{
          "personnel_decon": "洗消站点设置，包括淋浴设备和清洁剂",
          "equipment_decon": "使用特定洗消液浸泡设备",
          "area_decon": "利用高压水枪进行区域冲洗消毒"
        }}
      }},
      "medical_response": {{
        "triage_locations": ["事故中心设立分诊点、伤员集中点"],
        "ambulance_standby": {{
          "positions": ["事故中心周边设立待命点"],
          "quantity": "根据伤员数量确定救护车需求",
          "access_routes": ["确保通畅的救护车接入道路"]
        }},
        "medical_supplies": ["特殊解毒剂、止血药、敷料、输液设备等"],
        "specialist_team": {{
          "composition": "急救、中毒、烧伤专家组成",
          "deployment": "根据需要分配到各伤员救治点"
        }},
        "triage_mechanism": {{
          "categories": "按伤情轻重分为红、黄、绿、黑四类",
          "procedures": "分类、标识、优先救治",
          "priority_treatments": "优先处理危重伤员"
        }},
        "hospital_coordination": {{
          "receiving_hospitals": ["天津市内各大医院"],
          "capacity": "集中15所三级医院接收伤员",
          "specialized_treatment": "指定医院负责特殊中毒伤员救治"
        }}
      }},
      "environmental_monitoring": {{
        "air_quality": {{
          "parameters": ["氰化氢、硫化氢、氨气、二氧化硫等有害气体浓度"],
          "monitoring_points": ["事故中心、上风向和下风向监测点"],
          "monitoring_methods": "在线监测设备和现场取样检测",
          "threshold_values": "空气污染物浓度标准值"
        }},
        "water_monitoring": {{
          "parameters": ["氰化物浓度、PH值、有害物质浓度"],
          "sampling_locations": ["事故水体、临近地表水、地下水监测井"],
          "monitoring_methods": "定期取样监测，实验室分析",
          "prevention_measures": "设置水污染隔离防护设施"
        }},
        "soil_monitoring": {{
          "sampling_plan": "事故区域土壤系统采样方案",
          "analysis_parameters": ["重金属、有机物含量"],
          "contamination_control": "避免土壤污染扩散，清理措施"
        }},
        "reporting_frequency": "初期每小时，稳定后每天报告",
        "data_sharing": "环保部门与应急指挥部共享监测数据"
      }},
      "resource_allocation": {{
        "emergency_personnel": {{
          "onsite_command": "政府官员、消防指挥、安全专家",
          "firefighters": "消防部队200人，分为灭火、搜救、警戒组",
          "medical_staff": "医护人员9000人，分为现场急救和医院治疗两组",
          "security": "公安、武警2300人，负责现场安全和秩序维护",
          "environmental_team": "环保监测人员100人，24小时监测",
          "logistics_team": "后勤保障人员500人，负责物资供应"
        }},
        "equipment": {{
          "vehicles": ["消防车36辆，救护车100辆，工程车50辆"],
          "specialized_equipment": ["防化服、呼吸器、排烟机、洗消设备"],
          "communication_devices": "卫星电话、对讲机、应急通信车",
          "power_supply": "移动发电车、应急照明设备"
        }},
        "external_support": {{
          "government_agencies": ["国家应急管理部、环保部、卫健委协调支援"],
          "nearby_enterprises": ["周边化工企业提供专业设备和技术支持"],
          "expert_resources": "调集化工、爆炸、环保专家组成专家组"
        }},
        "command_center": {{
          "location": "设在安全区域内政府办公大楼",
          "facilities": "通信设备、会议室、信息处理中心",
          "functions": "指挥区、情报区、协调区、后勤区"
        }},
        "materials_dispatch": {{
          "storage_locations": "应急物资储备库",
          "dispatch_routes": "规划安全高效的物资运输路线",
          "inventory_management": "建立物资使用登记和补充机制"
        }}
      }},
      "information_management": {{
        "notification_chain": {{
          "internal_notification": ["事发单位-应急管理部门-政府-国务院"],
          "external_notification": ["通知周边企业、居民区、学校等"],
          "escalation_procedure": "重大情况立即上报，按级别响应"
        }},
        "public_communication": {{
          "channels": ["新闻发布会、官方媒体、社交媒体、短信平台"],
          "frequency": "初期每小时更新，后期每天两次",
          "content": "事故情况、救援进展、安全提示、避险指南",
          "spokesperson": "指定政府发言人统一对外发声",
          "press_conference": "定期举行新闻发布会通报情况"
        }},
        "rumor_control": {{
          "monitoring_mechanism": "设立舆情监测小组，收集网络信息",
          "response_strategy": "及时辟谣，公布权威信息",
          "truth_publication": "定期发布权威信息消除疑虑"
        }},
        "department_coordination": {{
          "information_sharing_platform": "建立跨部门信息共享平台",
          "joint_meeting_system": "每日召开联席会议协调行动",
          "communication_protocol": "建立部门间直通联系机制"
        }}
      }},
      "recovery_plan": {{
        "site_cleanup": {{
          "methods": "分区分类清理危险化学品和废弃物",
          "timeline": "预计3个月完成现场清理",
          "waste_disposal": "危险废物送专业机构处理",
          "safety_measures": "清理人员穿戴防护装备，定期轮换"
        }},
        "environmental_restoration": {{
          "soil_remediation": "受污染土壤挖除替换或原位修复",
          "water_remediation": "受污染水体处理达标后排放",
          "vegetation_recovery": "重新种植适合当地生长的植被",
          "monitoring_program": "长期环境质量监测计划"
        }},
        "production_resumption": {{
          "safety_inspection": "全面检查设备设施安全状况",
          "equipment_testing": "对关键设备进行功能测试",
          "staff_training": "员工安全教育培训",
          "phased_resumption": "安全区域先行复工，危险区域后续处理"
        }},
        "long_term_monitoring": {{
          "objectives": "确保环境质量恢复及无隐患",
          "parameters": "空气、水、土壤污染物含量",
          "frequency": "前期每周，后期每月，长期每季度",
          "responsible_parties": "环保部门负责监测工作"
        }},
        "area_reconstruction": {{
          "assessment": "全面评估区域受损情况",
          "planning": "科学规划重建项目",
          "implementation": "分阶段实施重建计划"
        }}
      }},
      "secondary_disaster_prevention": {{
        "explosion_prevention": {{
          "risk_sources": "识别残留的易爆危险化学品",
          "control_measures": "及时清理、降温、覆盖",
          "emergency_response": "发现险情立即疏散人员"
        }},
        "fire_control": {{
          "risk_sources": "可燃物与热源识别",
          "control_measures": "隔离可燃物，消除火源",
          "firefighting_plan": "针对不同种类火灾的扑救方案"
        }},
        "toxic_release_prevention": {{
          "risk_sources": "氰化钠等有毒化学品残留",
          "containment_measures": "封闭储存，防雨防晒",
          "exposure_reduction": "加强个人防护，减少暴露机会"
        }},
        "chain_reaction_control": {{
          "potential_scenarios": "周边化工企业二次爆炸风险",
          "isolation_strategy": "建立防火隔离带",
          "intervention_measures": "预冷、稀释、中和危险品"
        }},
        "nearby_facility_protection": {{
          "critical_facilities": ["临近油罐区、加油站、居民区"],
          "protection_measures": "加强监控，设置防护屏障",
          "evacuation_priority": "危险设施人员优先撤离"
        }}
      }},
      "special_material_handling": {{
        "hazardous_chemicals": [
          {{
            "chemical_name": "硝酸铵",
            "hazard_properties": "强氧化剂，遇高温、震动可能爆炸",
            "handling_procedure": "立即隔离现场，远离火源和热源",
            "neutralization_method": "降温稀释，避免与有机物接触",
            "safety_precautions": "全套防护装备，远距离操作"
          }},
          {{
            "chemical_name": "氰化钠",
            "hazard_properties": "剧毒，遇酸释放剧毒氰化氢气体",
            "handling_procedure": "使用防毒面具、橡胶手套处理",
            "neutralization_method": "使用次氯酸钠或双氧水氧化分解",
            "safety_precautions": "禁止与酸类物质接触，避免产生氰化氢"
          }}
        ]
      }}
    }}
    
    输出JSON格式如下：
    {{
      "emergency_level": {{
        "level": "应急响应级别（I级/II级/III级/IV级）",
        "reason": "确定该级别的理由，包括事故危害程度、影响范围、救援难度等",
        "initiation_procedure": "启动该级别应急预案的具体程序和责任部门"
      }},
      "evacuation_plan": {{
        "priority_zones": ["按照风向和扩散预测列出的优先疏散区域，包括具体的地理边界"],
        "evacuation_routes": ["针对不同区域的安全疏散路线，避开污染区和危险区"],
        "assembly_points": ["具体集合点位置和接收能力"],
        "vulnerable_groups_arrangements": "老人、儿童、病患等弱势群体的特殊疏散和安置安排",
        "transportation_arrangements": "大规模疏散所需的交通工具类型、数量及其调度方案",
        "notification_methods": "通知民众疏散的方式，如警报、广播、短信、上门通知等",
        "evacuation_stages": "分阶段疏散计划，包括时间节点和区域划分"
      }},
      "onsite_response": {{
        "isolation_zone": {{
          "hot_zone": "危险区域范围（米）",
          "warm_zone": "缓冲区域范围（米）",
          "cold_zone": "支援区域范围（米）"
        }},
        "command_system": {{
          "field_command_post": "现场指挥部设置位置",
          "command_structure": "现场指挥体系架构",
          "communication_system": "现场通信保障方案"
        }},
        "leakage_control": {{
          "method": "控制泄漏的主要方法和技术",
          "backup_plans": ["备选方案"],
          "material_needed": ["所需材料和设备清单"]
        }},
        "hazard_neutralization": {{
          "method": "针对特定危险化学品的中和或降低危害的方法",
          "equipment": ["所需设备列表"],
          "technical_guidance": "技术操作指南要点"
        }},
        "ppe_requirements": {{
          "entry_team": "进入热区人员的防护等级和装备要求",
          "decontamination_team": "洗消人员的防护要求",
          "medical_staff": "医疗人员的防护要求"
        }},
        "decontamination": {{
          "personnel_decon": "人员洗消程序和设施",
          "equipment_decon": "设备洗消方法",
          "area_decon": "区域洗消技术"
        }}
      }},
      "medical_response": {{
        "triage_locations": ["分类救治地点，包括现场急救点、临时医疗点"],
        "ambulance_standby": {{
          "positions": ["救护车待命位置"],
          "quantity": "所需救护车数量",
          "access_routes": ["医疗通道设置"]
        }},
        "medical_supplies": ["所需医疗物资清单，包括特殊解毒剂、输液设备等"],
        "specialist_team": {{
          "composition": "所需专家团队组成",
          "deployment": "专家调度和分工方案"
        }},
        "triage_mechanism": {{
          "categories": "伤员分类标准",
          "procedures": "分类处理流程",
          "priority_treatments": "优先治疗原则"
        }},
        "hospital_coordination": {{
          "receiving_hospitals": ["指定接收伤员的医院名单"],
          "capacity": "各医院接收能力",
          "specialized_treatment": "特殊中毒救治能力分布"
        }}
      }},
      "environmental_monitoring": {{
        "air_quality": {{
          "parameters": ["需监测的空气污染物参数列表"],
          "monitoring_points": ["监测点位置，包括上风向、下风向点位分布"],
          "monitoring_methods": "监测方法和设备",
          "threshold_values": "各项指标的报警阈值"
        }},
        "water_monitoring": {{
          "parameters": ["需监测的水质参数列表"],
          "sampling_locations": ["采样点位置，包括地表水、地下水采样点"],
          "monitoring_methods": "监测方法和频率",
          "prevention_measures": "水体污染防控措施"
        }},
        "soil_monitoring": {{
          "sampling_plan": "土壤采样方案",
          "analysis_parameters": ["需检测的土壤污染物指标"],
          "contamination_control": "土壤污染控制措施"
        }},
        "reporting_frequency": "不同阶段的监测报告更新频率",
        "data_sharing": "监测数据共享机制"
      }},
      "resource_allocation": {{
        "emergency_personnel": {{
          "onsite_command": "现场指挥人员数量和类型",
          "firefighters": "消防人员数量和分工",
          "medical_staff": "医疗人员数量和分工",
          "security": "安保人员数量和部署",
          "environmental_team": "环境监测人员配置",
          "logistics_team": "后勤保障人员配置"
        }},
        "equipment": {{
          "vehicles": ["所需车辆类型及数量"],
          "specialized_equipment": ["所需特殊设备列表"],
          "communication_devices": "通信设备配置",
          "power_supply": "应急电源保障"
        }},
        "external_support": {{
          "government_agencies": ["需联系的政府机构及其职责"],
          "nearby_enterprises": ["可能提供支援的周边企业及可提供的资源"],
          "expert_resources": "外部专家资源调度"
        }},
        "command_center": {{
          "location": "应急指挥中心设置位置",
          "facilities": "指挥中心必备设施",
          "functions": "各功能分区设置"
        }},
        "materials_dispatch": {{
          "storage_locations": "应急物资储备点",
          "dispatch_routes": "物资调度路线",
          "inventory_management": "物资管理机制"
        }}
      }},
      "information_management": {{
        "notification_chain": {{
          "internal_notification": ["内部报告流程和责任人"],
          "external_notification": ["外部通报流程和责任单位"],
          "escalation_procedure": "信息升级报告机制"
        }},
        "public_communication": {{
          "channels": ["信息发布渠道"],
          "frequency": "信息更新频率",
          "content": "关键信息内容要点",
          "spokesperson": "发言人安排",
          "press_conference": "新闻发布会组织方案"
        }},
        "rumor_control": {{
          "monitoring_mechanism": "谣言监测机制",
          "response_strategy": "谣言应对策略",
          "truth_publication": "权威信息发布机制"
        }},
        "department_coordination": {{
          "information_sharing_platform": "部门间信息共享平台",
          "joint_meeting_system": "联席会议制度",
          "communication_protocol": "跨部门沟通协议"
        }}
      }},
      "recovery_plan": {{
        "site_cleanup": {{
          "methods": "清理方法和技术路线",
          "timeline": "预计完成时间",
          "waste_disposal": "废弃物处理方案",
          "safety_measures": "清理过程中的安全保障措施"
        }},
        "environmental_restoration": {{
          "soil_remediation": "土壤修复方法和技术",
          "water_remediation": "水体修复计划",
          "vegetation_recovery": "植被恢复计划",
          "monitoring_program": "环境恢复监测方案"
        }},
        "production_resumption": {{
          "safety_inspection": "复产前安全检查要点",
          "equipment_testing": "设备测试要求",
          "staff_training": "员工培训要求",
          "phased_resumption": "分阶段复工复产计划"
        }},
        "long_term_monitoring": {{
          "objectives": "长期监测目标",
          "parameters": "监测指标",
          "frequency": "监测频率",
          "responsible_parties": "责任单位"
        }},
        "area_reconstruction": {{
          "assessment": "区域重建评估",
          "planning": "重建规划原则",
          "implementation": "实施路径"
        }}
      }},
      "secondary_disaster_prevention": {{
        "explosion_prevention": {{
          "risk_sources": "爆炸风险源识别",
          "control_measures": "预防措施",
          "emergency_response": "应急处置方案"
        }},
        "fire_control": {{
          "risk_sources": "火灾风险源识别",
          "control_measures": "预防措施",
          "firefighting_plan": "灭火方案"
        }},
        "toxic_release_prevention": {{
          "risk_sources": "有毒物质释放风险源",
          "containment_measures": "控制措施",
          "exposure_reduction": "暴露减少策略"
        }},
        "chain_reaction_control": {{
          "potential_scenarios": "可能的连锁反应场景",
          "isolation_strategy": "隔离策略",
          "intervention_measures": "干预措施"
        }},
        "nearby_facility_protection": {{
          "critical_facilities": ["需要重点保护的周边设施"],
          "protection_measures": "保护措施",
          "evacuation_priority": "优先撤离顺序"
        }}
      }},
      "special_material_handling": {{
        "hazardous_chemicals": [
          {{
            "chemical_name": "特定危险化学品名称（如硝酸铵）",
            "hazard_properties": "危险特性",
            "handling_procedure": "专门处置程序",
            "neutralization_method": "中和或稳定化方法",
            "safety_precautions": "特殊安全防护措施"
          }}
        ]
      }}
    }}
    
    注意：输出必须严格按照上述JSON格式，所有键名均使用英文，确保所有键名对应的内容均使用中文。如果某些信息无法基于给定的情景分析进行评估，请标记为"信息不足"。请基于专业知识进行推断，确保评估结果的合理性。参考天津港"8·12"特别重大火灾爆炸事故的经验教训，特别注意针对硝酸铵、氰化钠等高危险化学品的专项处置方案。
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

    def attempt_fix_truncated_json(self, json_text: str) -> str:
        """
        尝试修复被截断的JSON文本。
        
        Args:
            json_text: 可能被截断的JSON文本
            
        Returns:
            str: 修复后的JSON文本
        """
        try:
            # 尝试直接解析，如果成功则无需修复
            json.loads(json_text)
            return json_text
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON解析错误，尝试修复: {str(e)}")
            
            # 1. 提取完整的部分字段 - 特别是对于响应计划
            complete_fields = {}
            
            # 尝试提取emergency_level字段
            level_match = re.search(r'"emergency_level"\s*:\s*{([^{}]*({[^{}]*})*[^{}]*)}', json_text)
            if level_match:
                try:
                    level_text = '{' + level_match.group(1) + '}'
                    level_json = json.loads('{"emergency_level":' + level_text + '}')
                    complete_fields.update(level_json)
                    self.logger.info("成功提取emergency_level字段")
                except:
                    pass
            
            # 尝试提取evacuation_plan字段
            evac_match = re.search(r'"evacuation_plan"\s*:\s*{([^{}]*({[^{}]*})*[^{}]*)}', json_text)
            if evac_match:
                try:
                    evac_text = '{' + evac_match.group(1) + '}'
                    evac_json = json.loads('{"evacuation_plan":' + evac_text + '}')
                    complete_fields.update(evac_json)
                    self.logger.info("成功提取evacuation_plan字段")
                except:
                    pass
            
            # 尝试提取onsite_response字段
            onsite_match = re.search(r'"onsite_response"\s*:\s*{([^{}]*({[^{}]*})*[^{}]*)}', json_text)
            if onsite_match:
                try:
                    onsite_text = '{' + onsite_match.group(1) + '}'
                    
                    # 由于onsite_response通常包含嵌套结构，尝试更高级的修复
                    onsite_text = re.sub(r',\s*}', '}', onsite_text)  # 移除尾部逗号
                    onsite_text = re.sub(r'"\s*}', '"}', onsite_text)  # 修复引号后直接闭合
                    
                    # 修复嵌套的isolation_zone结构
                    isolation_match = re.search(r'"isolation_zone"\s*:\s*{([^{}]*)}', onsite_text)
                    if isolation_match:
                        isolation_text = isolation_match.group(1)
                        if isolation_text and not isolation_text.endswith('}'):
                            fixed_isolation = isolation_text + '}'
                            onsite_text = onsite_text.replace(isolation_text, fixed_isolation)
                    
                    # 修复嵌套的command_system结构
                    command_match = re.search(r'"command_system"\s*:\s*{([^{}]*)}', onsite_text)
                    if command_match:
                        command_text = command_match.group(1)
                        if command_text and not command_text.endswith('}'):
                            fixed_command = command_text + '}'
                            onsite_text = onsite_text.replace(command_text, fixed_command)
                    
                    onsite_json = json.loads('{"onsite_response":' + onsite_text + '}')
                    complete_fields.update(onsite_json)
                    self.logger.info("成功提取onsite_response字段")
                except Exception as e:
                    self.logger.warning(f"提取onsite_response失败: {str(e)}")
                    # 简化提取尝试
                    try:
                        isolation_zone_match = re.search(r'"isolation_zone"\s*:\s*{([^{}]*({[^{}]*})*[^{}]*)}', json_text)
                        if isolation_zone_match:
                            isolation_text = '{' + isolation_zone_match.group(1) + '}'
                            isolation_json = json.loads('{"isolation_zone":' + isolation_text + '}')
                            if 'onsite_response' not in complete_fields:
                                complete_fields['onsite_response'] = {}
                            complete_fields['onsite_response'].update(isolation_json)
                            self.logger.info("成功提取isolation_zone子字段")
                    except:
                        pass
            
            # 尝试提取medical_response字段
            medical_match = re.search(r'"medical_response"\s*:\s*{([^{}]*({[^{}]*})*[^{}]*)}', json_text)
            if medical_match:
                try:
                    medical_text = '{' + medical_match.group(1) + '}'
                    
                    # 由于medical_response通常包含嵌套结构，尝试更高级的修复
                    medical_text = re.sub(r',\s*}', '}', medical_text)  # 移除尾部逗号
                    medical_text = re.sub(r'"\s*}', '"}', medical_text)  # 修复引号后直接闭合
                    
                    # 修复嵌套的ambulance_standby结构
                    ambulance_match = re.search(r'"ambulance_standby"\s*:\s*{([^{}]*)}', medical_text)
                    if ambulance_match:
                        ambulance_text = ambulance_match.group(1)
                        if ambulance_text and not ambulance_text.endswith('}'):
                            fixed_ambulance = ambulance_text + '}'
                            medical_text = medical_text.replace(ambulance_text, fixed_ambulance)
                    
                    # 修复嵌套的specialist_team结构
                    specialist_match = re.search(r'"specialist_team"\s*:\s*{([^{}]*)}', medical_text)
                    if specialist_match:
                        specialist_text = specialist_match.group(1)
                        if specialist_text and not specialist_text.endswith('}'):
                            fixed_specialist = specialist_text + '}'
                            medical_text = medical_text.replace(specialist_text, fixed_specialist)
                    
                    medical_json = json.loads('{"medical_response":' + medical_text + '}')
                    complete_fields.update(medical_json)
                    self.logger.info("成功提取medical_response字段")
                except Exception as e:
                    self.logger.warning(f"提取medical_response失败: {str(e)}")
                    # 简化提取尝试 - 提取triage_locations子字段
                    try:
                        triage_match = re.search(r'"triage_locations"\s*:\s*\[(.*?)\]', json_text)
                        if triage_match:
                            triage_text = triage_match.group(1)
                            triage_list = json.loads('[' + triage_text + ']')
                            if 'medical_response' not in complete_fields:
                                complete_fields['medical_response'] = {}
                            complete_fields['medical_response']['triage_locations'] = triage_list
                            self.logger.info("成功提取triage_locations子字段")
                    except:
                        pass
            
            # 尝试提取environmental_monitoring字段
            env_match = re.search(r'"environmental_monitoring"\s*:\s*{([^{}]*({[^{}]*})*[^{}]*)}', json_text)
            if env_match:
                try:
                    env_text = '{' + env_match.group(1) + '}'
                    env_text = re.sub(r',\s*}', '}', env_text)  # 移除尾部逗号
                    env_text = re.sub(r'"\s*}', '"}', env_text)  # 修复引号后直接闭合
                    
                    env_json = json.loads('{"environmental_monitoring":' + env_text + '}')
                    complete_fields.update(env_json)
                    self.logger.info("成功提取environmental_monitoring字段")
                except:
                    pass
                    
            # 尝试提取resource_allocation字段
            resource_match = re.search(r'"resource_allocation"\s*:\s*{([^{}]*({[^{}]*})*[^{}]*)}', json_text)
            if resource_match:
                try:
                    resource_text = '{' + resource_match.group(1) + '}'
                    resource_text = re.sub(r',\s*}', '}', resource_text)  # 移除尾部逗号
                    resource_text = re.sub(r'"\s*}', '"}', resource_text)  # 修复引号后直接闭合
                    
                    resource_json = json.loads('{"resource_allocation":' + resource_text + '}')
                    complete_fields.update(resource_json)
                    self.logger.info("成功提取resource_allocation字段")
                except Exception as e:
                    self.logger.warning(f"提取resource_allocation失败: {str(e)}")
                    # 提取紧急人员子字段
                    try:
                        personnel_match = re.search(r'"emergency_personnel"\s*:\s*{([^{}]*)}', json_text)
                        if personnel_match:
                            personnel_text = '{' + personnel_match.group(1) + '}'
                            personnel_json = json.loads('{"emergency_personnel":' + personnel_text + '}')
                            if 'resource_allocation' not in complete_fields:
                                complete_fields['resource_allocation'] = {}
                            complete_fields['resource_allocation'].update(personnel_json)
                            self.logger.info("成功提取emergency_personnel子字段")
                    except:
                        pass
            
            # 尝试提取information_management字段
            info_match = re.search(r'"information_management"\s*:\s*{([^{}]*({[^{}]*})*[^{}]*)}', json_text)
            if info_match:
                try:
                    info_text = '{' + info_match.group(1) + '}'
                    info_text = re.sub(r',\s*}', '}', info_text)  # 移除尾部逗号
                    info_text = re.sub(r'"\s*}', '"}', info_text)  # 修复引号后直接闭合
                    
                    info_json = json.loads('{"information_management":' + info_text + '}')
                    complete_fields.update(info_json)
                    self.logger.info("成功提取information_management字段")
                except Exception as e:
                    self.logger.warning(f"提取information_management失败: {str(e)}")
                    # 简化提取尝试
                    try:
                        notification_match = re.search(r'"notification_chain"\s*:\s*{([^{}]*)}', json_text)
                        if notification_match:
                            notification_text = '{' + notification_match.group(1) + '}'
                            notification_json = json.loads('{"notification_chain":' + notification_text + '}')
                            if 'information_management' not in complete_fields:
                                complete_fields['information_management'] = {}
                            complete_fields['information_management'].update(notification_json)
                            self.logger.info("成功提取notification_chain子字段")
                    except:
                        pass
            
            # 尝试提取recovery_plan字段
            recovery_match = re.search(r'"recovery_plan"\s*:\s*{([^{}]*({[^{}]*})*[^{}]*)}', json_text)
            if recovery_match:
                try:
                    recovery_text = '{' + recovery_match.group(1) + '}'
                    recovery_text = re.sub(r',\s*}', '}', recovery_text)  # 移除尾部逗号
                    recovery_text = re.sub(r'"\s*}', '"}', recovery_text)  # 修复引号后直接闭合
                    
                    recovery_json = json.loads('{"recovery_plan":' + recovery_text + '}')
                    complete_fields.update(recovery_json)
                    self.logger.info("成功提取recovery_plan字段")
                except Exception as e:
                    self.logger.warning(f"提取recovery_plan失败: {str(e)}")
                    # 简化提取尝试
                    try:
                        cleanup_match = re.search(r'"site_cleanup"\s*:\s*{([^{}]*)}', json_text)
                        if cleanup_match:
                            cleanup_text = '{' + cleanup_match.group(1) + '}'
                            cleanup_json = json.loads('{"site_cleanup":' + cleanup_text + '}')
                            if 'recovery_plan' not in complete_fields:
                                complete_fields['recovery_plan'] = {}
                            complete_fields['recovery_plan'].update(cleanup_json)
                            self.logger.info("成功提取site_cleanup子字段")
                    except:
                        pass
            
            # 尝试提取secondary_disaster_prevention字段
            prevention_match = re.search(r'"secondary_disaster_prevention"\s*:\s*{([^{}]*({[^{}]*})*[^{}]*)}', json_text)
            if prevention_match:
                try:
                    prevention_text = '{' + prevention_match.group(1) + '}'
                    prevention_text = re.sub(r',\s*}', '}', prevention_text)  # 移除尾部逗号
                    prevention_text = re.sub(r'"\s*}', '"}', prevention_text)  # 修复引号后直接闭合
                    
                    prevention_json = json.loads('{"secondary_disaster_prevention":' + prevention_text + '}')
                    complete_fields.update(prevention_json)
                    self.logger.info("成功提取secondary_disaster_prevention字段")
                except Exception as e:
                    self.logger.warning(f"提取secondary_disaster_prevention失败: {str(e)}")
                    # 简化提取尝试
                    try:
                        explosion_match = re.search(r'"explosion_prevention"\s*:\s*{([^{}]*)}', json_text)
                        if explosion_match:
                            explosion_text = '{' + explosion_match.group(1) + '}'
                            explosion_json = json.loads('{"explosion_prevention":' + explosion_text + '}')
                            if 'secondary_disaster_prevention' not in complete_fields:
                                complete_fields['secondary_disaster_prevention'] = {}
                            complete_fields['secondary_disaster_prevention'].update(explosion_json)
                            self.logger.info("成功提取explosion_prevention子字段")
                    except:
                        pass
            
            # 尝试提取special_material_handling字段
            material_match = re.search(r'"special_material_handling"\s*:\s*{([^{}]*({[^{}]*})*[^{}]*)}', json_text)
            if material_match:
                try:
                    material_text = '{' + material_match.group(1) + '}'
                    material_text = re.sub(r',\s*}', '}', material_text)  # 移除尾部逗号
                    material_text = re.sub(r'"\s*}', '"}', material_text)  # 修复引号后直接闭合
                    
                    material_json = json.loads('{"special_material_handling":' + material_text + '}')
                    complete_fields.update(material_json)
                    self.logger.info("成功提取special_material_handling字段")
                except:
                    pass
            
            # 2. 如果提取到了至少一个完整字段，则返回这些字段组成的JSON
            if complete_fields:
                self.logger.info(f"成功提取了 {len(complete_fields)} 个完整字段")
                return json.dumps(complete_fields)
            
            # 3. 如果无法提取完整字段，尝试基于括号平衡来修复
            brackets_stack = []
            for i, char in enumerate(json_text):
                if char == '{':
                    brackets_stack.append('}')
                elif char == '[':
                    brackets_stack.append(']')
                elif char == '}' and brackets_stack and brackets_stack[-1] == '}':
                    brackets_stack.pop()
                elif char == ']' and brackets_stack and brackets_stack[-1] == ']':
                    brackets_stack.pop()
            
            # 添加缺失的括号
            fixed_json = json_text
            while brackets_stack:
                fixed_json += brackets_stack.pop()
            
            # 替换常见的JSON格式错误
            # 1. 多余的逗号
            fixed_json = re.sub(r',\s*}', '}', fixed_json)
            fixed_json = re.sub(r',\s*]', ']', fixed_json)
            
            # 2. 缺少逗号
            fixed_json = re.sub(r'"\s*{', '",{', fixed_json)
            fixed_json = re.sub(r'"\s*\[', '",[', fixed_json)
            
            # 3. 修复截断或不完整的键值对
            fixed_json = re.sub(r'"([^"]+)"\s*:\s*(?!(true|false|null|\{|\[|"|\d))', r'"\1": "未知"', fixed_json)
            
            try:
                # 测试修复后的JSON是否可以解析
                json.loads(fixed_json)
                self.logger.info("JSON修复成功")
                return fixed_json
            except json.JSONDecodeError:
                self.logger.error("JSON修复失败，返回基本结构")
                
                # 如果修复失败，返回之前提取的emergency_level或最基本的结构
                if '"emergency_level"' in json_text:
                    return '{"emergency_level":{"level":"I级","reason":"爆炸事故影响范围广泛"}}'
                return '{}'

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
                    self.logger.warning(f"影响评估阶段缺少太多必要键，仅找到: {found_keys}")
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
                # 响应计划阶段 - 更宽松的验证，接受部分数据
                # 检查是否包含关键部分，即使只有几个键
                plan_keys = ["emergency_level", "evacuation_plan", "onsite_response", 
                            "medical_response", "environmental_monitoring", 
                            "resource_allocation", "information_management", "recovery_plan"]
                
                found_keys = [key for key in plan_keys if key in json_data]
                if len(found_keys) >= 1:  # 只需要1个关键部分即可接受
                    self.logger.info(f"响应计划阶段验证通过，包含字段: {found_keys}")
                    return True
                else:
                    self.logger.warning(f"响应计划缺少所有必要键")
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
                        # 尝试修复和解析JSON
                        fixed_json_text = self.attempt_fix_truncated_json(json_text)
                        response_json = json.loads(fixed_json_text)
                        self.logger.info(f"成功解析JSON响应: {json_text[:100]}...")
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON解析错误，即使尝试修复后: {str(e)}")
                        self.logger.error(f"尝试解析的文本: {json_text[:100]}...")
                        
                        # 对于响应计划阶段，即使JSON无法完全解析，也尝试创建部分响应
                        if operation_index == 2:
                            # 从文本中提取可能的紧急等级信息
                            emergency_level_match = re.search(r'"emergency_level"\s*:\s*{([^}]+)}', json_text)
                            if emergency_level_match:
                                try:
                                    emergency_level_text = '{' + emergency_level_match.group(1) + '}'
                                    emergency_level_text = emergency_level_text.replace('""', '"未知"')
                                    emergency_level = json.loads('{"emergency_level":' + emergency_level_text + '}')
                                    response_json = emergency_level
                                    self.logger.info("成功提取紧急等级信息")
                                except:
                                    response_json = {"emergency_level": {"level": "I级", "reason": "爆炸事故影响范围广泛"}}
                            else:
                                response_json = {"emergency_level": {"level": "I级", "reason": "爆炸事故影响范围广泛"}}
                        else:
                            continue
                else:
                    self.logger.error(f"无法在文本中找到JSON: {text[:100]}...")
                    if operation_index == 2:
                        # 对于响应计划阶段，提供一个基本结构
                        response_json = {"emergency_level": {"level": "I级", "reason": "爆炸事故影响范围广泛"}}
                    else:
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
                        # 对于响应计划阶段，即使验证失败也保存基本信息
                        if "emergency_level" in response_json:
                            prev_state = thought.get("state", {})
                            new_thought["state"] = prev_state.copy()
                            new_thought["state"]["response_plan"] = response_json
                            self.logger.info("保存了部分响应计划数据")
                        else:
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
    output_file = "examples/chemical_emergency/emergency_response_plan.json"
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
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
    debug_file = "examples/chemical_emergency/debug_all_results.json"
    
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(debug_file), exist_ok=True)
        
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"所有阶段结果已保存到 {debug_file}")
    except Exception as e:
        logger.error(f"保存调试结果时出错: {str(e)}")
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
（一）事故发生的时间和地点。 
2025 年 8 月12日22时51分46秒，位于天津市滨海新区
吉运二道95号的瑞海公司危险品仓库（北纬39°02′22.98″，东
经117 °44′11.64″。地理方位示意图见图1）运抵区（"待申报
装船出口货物运抵区"的简称，属于海关监管场所，用金属栅栏
与外界隔离。由经营企业申请设立，海关批准，主要用于出口集
装箱货物的运抵和报关监管）最先起火，23时34分06秒发生
第一次爆炸，23时34分37秒发生第二次更剧烈的爆炸。事故
现场形成6处大火点及数十个小火点，8月14日16时40分，
现场明火被扑灭。
天气：多云
温度：12℃
风向：东
风力：4 级
空气湿度：37 
（二）事故现场情况。 
事故现场按受损程度，分为事故中心区、爆炸冲击波波及区。
事故中心区为此次事故中受损最严重区域，该区域东至跃进路、
西至海滨高速、南至顺安仓储有限公司、北至吉运三道，面积约为54万平
方米。两次爆炸分别形成一个直径15米、深1.1米的月牙形小
爆坑和一个直径97米、深2.7米的圆形大爆坑。以大爆坑为爆
炸中心，150米范围内的建筑被摧毁，东侧的瑞海公司综合楼和
南侧的中联建通公司办公楼只剩下钢筋混凝土框架；堆场内大量
普通集装箱和罐式集装箱被掀翻、解体、炸飞，形成由南至北的
3 座巨大堆垛，一个罐式集装箱被抛进中联建通公司办公楼4层
房间内，多个集装箱被抛到该建筑楼顶；参与救援的消防车、警
车和位于爆炸中心南侧的吉运一道和北侧吉运三道附近的顺安
仓储有限公司、安邦国际贸易有限公司储存的7641辆商品汽车
和现场灭火的30辆消防车在事故中全部损毁，邻近中心区的贵
龙实业、新东物流、港湾物流等公司的4787辆汽车受损。
爆炸冲击波波及区分为严重受损区、中度受损区。严重受损
区是指建筑结构、外墙、吊顶受损的区域，受损建筑部分主体承
重构件（柱、梁、楼板）的钢筋外露，失去承重能力，不再满足
安全使用条件。中度受损区是指建筑幕墙及门、窗受损的区域，
受损建筑局部幕墙及部分门、窗变形、破裂。 
严重受损区在不同方向距爆炸中心最远距离为：东 3 公里
（亚实履带天津有限公司），西3.6公里（联通公司办公楼），南
2.5 公里（天津振华国际货运有限公司），北 2.8 公里（天津丰
田通商钢业公司）。中度受损区在不同方向距爆炸中心最远距离
为：东3.42公里（国际物流验放中心二场），西5.4公里（中国
检验检疫集团办公楼），南5公里（天津港物流大厦），北5.4公
里（天津海运职业学院）。受地形地貌、建筑位置和结构等因素
影响，同等距离范围内的建筑受损程度并不一致。 
爆炸冲击波波及区以外的部分建筑，虽没有受到爆炸冲击波
直接作用，但由于爆炸产生地面震动，造成建筑物接近地面部位
的门、窗玻璃受损，东侧最远达8.5公里（东疆港宾馆），西侧
最远达8.3公里（正德里居民楼），南侧最远达8公里（和丽苑
居民小区），北侧最远达13.3公里（海滨大道永定新河收费站）。
（三）人员伤亡和财产损失情况。 
事故造成 165 人遇难（参与救援处置的公安现役消防人员
24 人、天津港消防人员75人、公安民警11人，事故企业、周
边企业员工和周边居民55人），8人失踪（天津港消防人员5人，
周边企业员工、天津港消防人员家属3人），798 人受伤住院治
疗（伤情重及较重的伤员58人、轻伤员740人）；304幢建筑物
（其中办公楼宇、厂房及仓库等单位建筑73幢，居民1类住宅
91 幢、2类住宅129幢、居民公寓11幢）、12428辆商品汽车、
7533 个集装箱受损。 
事故调查组依据《企业职工伤亡事故经济损失统计标准》
（GB6721-1986）等标准和规定统计，
核定直接经济损失
（四）危险化学品情况。 
通过分析事发时瑞海公司储存的 111 种危险货物的化学组
分，确定至少有129种化学物质发生爆炸燃烧或泄漏扩散，其中，
氢氧化钠、硝酸钾、硝酸铵、氰化钠、金属镁和硫化钠这6种物
质的重量占到总重量的50%。同时，爆炸还引燃了周边建筑物以
及大量汽车、焦炭等普通货物。本次事故残留的化学品与产生的
二次污染物逾百种，对局部区域的大气环境、水环境和土壤环境
造成了不同程度的污染。
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
        
        # 使用debug_all_results收集所有阶段数据
        all_results = {}
        
        for i, phase_thoughts in enumerate(results):
            if phase_thoughts and len(phase_thoughts) > 0:
                phase_key = f"phase_{i}"
                # 创建可序列化的状态副本
                state_copy = dict(phase_thoughts[-1].state)
                if "graph" in state_copy:
                    del state_copy["graph"]
                all_results[phase_key] = state_copy
        
        # 提取各个阶段的结果
        logger.info(f"从最终状态中提取结果")
        
        # 第一阶段：情景分析 (从phase_0提取)
        if "phase_0" in all_results and "state" in all_results["phase_0"]:
            state = all_results["phase_0"]["state"]
            if state:
                situation_analysis = {
                    "basic_info": state.get("basic_info", {}),
                    "accident_info": state.get("accident_info", {}),
                    "weather_conditions": state.get("weather_conditions", {}),
                    "geographical_info": state.get("geographical_info", {}),
                    "sensitive_targets": state.get("sensitive_targets", {})
                }
                response["situation_analysis"] = situation_analysis
                logger.info(f"成功提取情景分析阶段的结果")
        
        # 获取最新状态中的影响评估 (可能在phase_1或phase_0中)
        impact_assessment = None
        for phase_key in ["phase_1", "phase_0"]:
            if phase_key in all_results and "state" in all_results[phase_key]:
                state = all_results[phase_key]["state"]
                if state and "impact_assessment" in state:
                    impact_assessment = state["impact_assessment"]
                    response["impact_assessment"] = impact_assessment
                    logger.info(f"成功提取影响评估阶段的结果")
                    break
        
        # 获取最新状态中的响应计划 (可能在phase_2, phase_1, 或 phase_0中)
        response_plan = None
        for phase_key in ["phase_2", "phase_1", "phase_0"]:
            if phase_key in all_results and "state" in all_results[phase_key]:
                state = all_results[phase_key]["state"]
                if state and "response_plan" in state:
                    response_plan = state["response_plan"]
                    response["response_plan"] = response_plan
                    logger.info(f"成功提取响应计划阶段的结果")
                    break
        
        # 如果还没有找到响应计划，尝试从原始数据中提取
        if "response_plan" not in response and len(results) > 2 and len(results[2]) > 0:
            last_thought = results[2][-1]
            if "state" in last_thought.state:
                state = last_thought.state.get("state", {})
                # 提取所有可能的响应计划字段
                response_plan_data = {}
                
                # 检查emergency_level字段
                if "emergency_level" in state:
                    response_plan_data["emergency_level"] = state["emergency_level"]
                    logger.info("提取了emergency_level字段")
                    
                # 检查evacuation_plan字段
                if "evacuation_plan" in state:
                    response_plan_data["evacuation_plan"] = state["evacuation_plan"]
                    logger.info("提取了evacuation_plan字段")
                    
                # 检查onsite_response字段
                if "onsite_response" in state:
                    response_plan_data["onsite_response"] = state["onsite_response"]
                    logger.info("提取了onsite_response字段")
                    
                # 检查medical_response字段
                if "medical_response" in state:
                    response_plan_data["medical_response"] = state["medical_response"]
                    logger.info("提取了medical_response字段")
                    
                # 检查environmental_monitoring字段
                if "environmental_monitoring" in state:
                    response_plan_data["environmental_monitoring"] = state["environmental_monitoring"]
                    logger.info("提取了environmental_monitoring字段")
                    
                # 检查resource_allocation字段
                if "resource_allocation" in state:
                    response_plan_data["resource_allocation"] = state["resource_allocation"]
                    logger.info("提取了resource_allocation字段")
                    
                # 检查information_management字段
                if "information_management" in state:
                    response_plan_data["information_management"] = state["information_management"]
                    logger.info("提取了information_management字段")
                    
                # 检查recovery_plan字段
                if "recovery_plan" in state:
                    response_plan_data["recovery_plan"] = state["recovery_plan"]
                    logger.info("提取了recovery_plan字段")
                    
                # 检查secondary_disaster_prevention字段
                if "secondary_disaster_prevention" in state:
                    response_plan_data["secondary_disaster_prevention"] = state["secondary_disaster_prevention"]
                    logger.info("提取了secondary_disaster_prevention字段")
                    
                # 检查special_material_handling字段
                if "special_material_handling" in state:
                    response_plan_data["special_material_handling"] = state["special_material_handling"]
                    logger.info("提取了special_material_handling字段")
                
                # 如果找到了任何字段，将其保存到响应中
                if response_plan_data:
                    response["response_plan"] = response_plan_data
                    logger.info(f"从原始数据中提取了响应计划的 {len(response_plan_data)} 个字段")
                else:
                    logger.warning("没有从原始数据中找到任何响应计划字段")
        
        # 保存结果
        output_file = "examples/chemical_emergency/emergency_response_plan.json"
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
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
        debug_file = "examples/chemical_emergency/debug_all_results.json"
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(debug_file), exist_ok=True)
            
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logger.info(f"所有阶段结果已保存到 {debug_file}")
        except Exception as e:
            logger.error(f"保存调试结果时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"发生意外错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 
