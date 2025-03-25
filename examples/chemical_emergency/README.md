# 化工事故应急响应决策支持系统

基于Graph of Thoughts (GOT)的化工事故应急响应决策支持系统，通过多阶段分析和推理，为化工事故应急响应提供智能决策支持。

## 系统功能

系统主要包含三个阶段的分析：

1. **事故情况分析**
   - 基础信息提取（时间、地点、企业信息）
   - 事故类型和发展过程分析
   - 气象条件分析
   - 地理环境分析
   - 周边敏感目标识别
   - 涉事化学品信息分析

2. **影响评估**
   - 扩散预测（范围、方向、速度）
   - 人员影响评估（伤亡预测、疏散范围）
   - 环境影响评估（大气、水体、土壤）
   - 次生灾害风险评估
   - 社会影响评估

3. **应急响应方案生成**
   - 应急响应等级和指挥体系
   - 人员疏散方案
   - 现场处置方案
   - 医疗救援方案
   - 环境监测与防护方案
   - 应急资源调配方案
   - 信息发布方案
   - 恢复重建计划

## 安装要求

- Python 3.8+
- 依赖包：
  ```
  graph-of-thoughts
  deepseek
  ```

## 使用方法

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置API密钥：
   - 在环境变量中设置DeepSeek API密钥：
     ```bash
     export DEEPSEEK_API_KEY=your_api_key
     ```

3. 运行系统：
   ```bash
   python chemical_emergency.py
   ```

4. 查看结果：
   - 系统将生成`emergency_response_plan.json`文件，包含完整的分析结果
   - 运行日志保存在`chemical_emergency.log`文件中

## 输入格式

系统接受文本格式的事故描述信息，应包含以下要素：
- 事故发生时间
- 事故地点和企业信息
- 事故类型和基本情况
- 气象条件
- 地理环境信息
- 涉事化学品信息

示例：
```text
2024年3月15日14时30分，天嘉宜化工有限公司发生化学品泄漏事故。
事故发生时，天气晴朗，东南风3级，温度25℃，相对湿度60%。
工厂位于化工园区内，距离最近的居民区2.5公里，周边500米内有其他3家化工企业。
泄漏物质为苯，预计泄漏量约5吨。现场已发现明显刺激性气味，部分工人出现呼吸不适症状。
```

## 输出格式

系统输出为JSON格式，包含三个主要部分：
- 事故情况分析结果
- 影响评估结果
- 应急响应方案

输出文件示例：
```json
{
  "situation_analysis": {
    "basic_info": {...},
    "accident_info": {...},
    "weather_conditions": {...},
    "geographical_info": {...},
    "sensitive_targets": {...}
  },
  "impact_assessment": {
    "dispersion_prediction": {...},
    "population_impact": {...},
    "environmental_impact": {...},
    "secondary_disasters": {...},
    "social_impact": {...}
  },
  "response_plan": {
    "emergency_level": {...},
    "evacuation_plan": {...},
    "onsite_response": {...},
    "medical_response": {...},
    "environmental_monitoring": {...},
    "resource_allocation": {...},
    "information_management": {...},
    "recovery_plan": {...}
  }
}
```

## 注意事项

1. 系统生成的应急响应方案仅供参考，最终决策需要专业人员确认。
2. 确保输入信息的准确性和完整性，这直接影响分析结果的质量。
3. 定期更新和维护系统，确保应急预案符合最新的安全规范和标准。
4. 建议在使用前进行充分的测试和验证。

## 贡献指南

欢迎提交问题报告和改进建议。如需贡献代码：
1. Fork 本仓库
2. 创建新的分支
3. 提交更改
4. 发起 Pull Request

## 许可证

本项目采用 BSD 许可证。详见 LICENSE 文件。 