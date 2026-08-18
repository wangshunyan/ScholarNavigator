# 赛题要求摘要

## 任务目标

构建端到端学术论文智能搜索系统，针对自然语言描述的复杂科研查询完成查询理解、多策略检索、论文排序和结果归纳。

## 四项核心功能

1. **查询理解与分解**：识别研究意图、实体、方法和领域约束，将复杂问题拆成可检索子查询，并进行查询改写。
2. **自主搜索策略迭代**：使用大模型规划检索词、过滤低相关结果，并根据已检索论文动态调整关键词和检索方向。
3. **论文综合排序**：依据标题、摘要及可用全文信息进行细粒度相关性判断，输出排序后的论文列表。
4. **搜索结果归纳**：按用户意图整理结果，并以列表、关系图等结构化形式展示。

## 技术要求

- 可使用开源或商业大模型，提交材料需注明模型及版本；官方鼓励采用可复现的开源模型。
- 至少接入一种学术搜索 API，例如 Semantic Scholar、OpenAlex 或 PubMed。
- 方案需记录并控制 API 调用次数、Token 消耗和端到端延迟。

## 评分指标及权重

自动评分内部权重：

| 指标 | 权重 | 关注点 |
| --- | ---: | --- |
| F1 Score | 70% | 精确率与召回率的平衡 |
| 运行效率 | 20% | API 调用、Token 消耗、端到端延迟 |
| 回复结构化 | 10% | 列表、关系图等结构化结果 |

## 成绩构成

| 组成 | 权重 |
| --- | ---: |
| 公开测试集自动评分 | 30% |
| 隐藏测试集自动评分 | 30% |
| 创新性专家评分 | 15% |
| 方案落地可行性专家评分 | 15% |
| 算法泛化性专家评分 | 10% |

## 官方参考数据集

- [PaSa](https://github.com/bytedance/pasa)：包含 AutoScholarQuery、RealScholarQuery 等学术搜索数据。
- [AstaBench](https://github.com/allenai/asta-bench)：科研智能体评测套件。

## 官方参考系统

- [PaSa](https://github.com/bytedance/pasa)
- [SPAR](https://github.com/zjunlp/SPAR)
- [Ai2 Paper Finder](https://github.com/allenai/asta-paper-finder)
- [PaperQA2](https://github.com/Future-House/paper-qa)
