# 纽约出租车需求预测 —— 端到端 MLOps 平台

[English](README.md) | 简体中文

> 一个生产级系统，预测**纽约黄色出租车的出行需求**（按天和按小时），并按照真实机器学习团队的方式交付：无数据泄漏的建模、经过校准的风险区间、带自动晋升门禁的模型注册表、闭环的漂移重训练、加固的推理 API、Kubernetes 部署，以及包含安全扫描和镜像签名的六阶段 CI/CD 流水线。

<p>
<a href="https://github.com/hastikacheddy/nyc-taxi-demand-forecasting/actions/workflows/mlops_pipeline.yaml"><img alt="CI" src="https://github.com/hastikacheddy/nyc-taxi-demand-forecasting/actions/workflows/mlops_pipeline.yaml/badge.svg"></a>
<img alt="Python" src="https://img.shields.io/badge/python-3.11-blue">
<img alt="Tests" src="https://img.shields.io/badge/tests-109%20passing-brightgreen">
<img alt="Coverage gate" src="https://img.shields.io/badge/coverage%20gate-%E2%89%A570%25-green">
<img alt="Security" src="https://img.shields.io/badge/security-Bandit%20%7C%20Semgrep%20%7C%20Trivy-orange">
<img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

这不是一个附带 `model.pkl` 的 Notebook，而是围绕模型构建的**运维系统**——也就是模型训练完成之后、占机器学习工作 90% 的那部分。它基于真实数据构建，并按照 MLOps 成熟度框架进行评估。

---

## 目录
- [结果](#结果) · [为什么是生产级](#为什么是生产级而不是一个-notebook) · [架构](#架构)
- [功能](#功能) · [值得一提的工程决策](#值得一提的工程决策) · [快速开始](#快速开始) · [仓库结构](#仓库结构)

---

## 结果

在 2024 年 4 个月、约 **1300 万次真实行程**的 TLC 黄色出租车数据上进行样本外评估（按时段聚合为需求量）。评分采用严格按时间顺序的 **80/20 划分**——模型从未见过用于评估它的那段末尾数据。运行 `python scripts/evaluate.py` 即可端到端复现。

| 模型 | 测试窗口 | MAE | MAPE | 99% 区间覆盖率 |
|--------|-------------|----:|-----:|----------------------:|
| **按天** | 23 天 | 约 6,200 次行程 | **5.4 %** | 95.7 % |
| **按小时** | 24 天（576 小时） | 约 260 次行程 | **8.1 %** | 100 % |

99% 风险价值（VaR）区间经过**校准**：实际覆盖率达到或高于 99% 目标，同时区间宽度合理（每日区间约比点预测高 12–15%）。对 `2024-05-01` 的下一步生产预测为：**每天 117,078 次行程**（99% 运力目标为 134,564）；高峰小时目标为**每小时 2,592 次行程**。

![预测值与实际值对比](docs/forecast_vs_actual.png)

*上图：每日需求——点预测（虚线）在 99% VaR 区间内紧密跟随实际值。下图：测试期最后 7 天的每小时需求——模型完整捕捉了通勤节律（深夜低谷、早晚高峰），误差约 8%。*

---

## 为什么是*生产级*，而不是一个 Notebook

本仓库最重要的一项工程决策，也是我在面试中会首先介绍的：

> **原始 Notebook 中的模型其实是“现时估计（nowcast）”，而不是“预测（forecast）”——我发现了这个问题。**

原 Notebook 的特征中包含 `LagDelta = Volume − Lag_n`，其中含有*当前*时段的真实值。这会带来非常漂亮的样本内 MAE（约 13），但属于**数据泄漏**：模型只能“预测”那些真实值已经已知的时段。如果直接照搬上线，就会得到一个演示时表现出色、在生产中却毫无用处的模型。

因此我另外构建了一个**无泄漏的预测器**（[`src/forecasting/forecaster.py`](src/forecasting/forecaster.py)）：针对目标时段 *t* 的每一个特征都严格只使用 *t* **之前**的数据计算（`Lag1`、季节性滞后、*仅基于过去*的动量和滚动统计量，以及可以提前得知的日历特征）。经过诚实的样本外重新评估，得到的就是上面 **5.4% 的每日 MAPE**——一个在生产中真正值得信赖的数字。原始 Notebook 的逻辑原封不动地保留，新的预测器是增量添加的。

这正是*“我训练了一个模型”*与*“我理解模型为什么会在生产中失效”*之间的区别。

---

## 架构

```
 TLC 行程记录（公开 Parquet，约 1300 万行）
      │  data_cleaning_dag  (@daily)
      ▼  传感器 → 导入 + 清洗（UTC，剔除无效/越界数据）→ pandera 质量门禁
 data/{daily,hourly}_demand.csv   （各时段行程数，由 DVC 追踪）
      │
      ├─►  weekly_training_dag  (@weekly)
      │      校验 → 训练 LightGBM → GARCH σ → 冠军-挑战者门禁
      │      → MLflow 注册表 @champion   （+ 模型卡、SHA-256 完整性标签）
      │
      ├─►  daily / hourly_inference_dag
      │      传感器 → 预测下一时段 → 99% VaR 区间 → 影子日志（幂等 upsert）
      │
      └─►  monitoring_dag  (@daily)
             将已到期的预测与实际值关联 → 实际 MAE 与覆盖率
             → 检测到概念漂移时，自动触发重训练      ← 闭环
                                   │
 推理 API（FastAPI）───────────────┘  加载 @champion，提供 /predict 与 /metrics
      │   API 密钥认证 · 限流 · Prometheus 遥测 · 异常输出防护
      ▼
 Kubernetes（非 root、只读文件系统、HPA 2→6）· MinIO/S3 制品存储 · Prometheus+Grafana
```

> **理解整个仓库只需一个心智模型：_Airflow 负责指挥，`src/` 负责演奏。_**
> DAG 只决定*何时*执行、*以什么顺序*执行；真正的逻辑都在 `src/` 中，无论有没有 Airflow 都以完全相同的方式运行——这正是 **109 个测试**能够直接覆盖它的原因。

完整设计、每一项安全防护措施（以及*为什么*需要它）和运维手册，请参阅 **[ARCHITECTURE_AND_HANDOVER.md](ARCHITECTURE_AND_HANDOVER.md)**。

---

## 功能

### 🧠 建模与不确定性
| 内容 | 位置 |
|---|---|
| 无泄漏的单步 LightGBM 预测器（按天 + 按小时） | [`forecasting/forecaster.py`](src/forecasting/forecaster.py) |
| **GARCH(1,1) + 1 万次蒙特卡洛模拟的风险价值（VaR）**，生成校准后的 99% 缓冲 | [`inference/risk.py`](src/inference/risk.py) |
| **按小时的条件波动率 σ**——高峰时段更宽、深夜设下限（捕捉单一 GARCH σ 无法体现的异方差） | [`inference/risk.py`](src/inference/risk.py) |
| 可复现的样本外评估框架与图表 | [`scripts/evaluate.py`](scripts/evaluate.py) |

### 🔁 MLOps 生命周期
| 内容 | 位置 |
|---|---|
| 带**冠军-挑战者晋升门禁**的 MLflow 注册表——重训练的模型只有在留出集 MAE *优于*现任模型时才会上线，否则保留为 `@challenger` | [`common/promotion.py`](src/common/promotion.py) |
| **闭环漂移检测**——将到期预测与实际值关联，实际 MAE 上升 15% 即判定概念漂移，并自动触发重训练 | [`monitoring/scoring.py`](src/monitoring/scoring.py) |
| 基于 Evidently 的特征分布漂移检测 | [`monitoring/drift_detector.py`](src/monitoring/drift_detector.py) |
| Feast 特征存储（离线 → 在线物化） | [`feature_repo/`](feature_repo/) |
| 每个版本记录模型卡 / 治理元数据 | [`governance/cards.py`](src/governance/cards.py) |
| DVC 数据版本管理 | [`.dvc/`](.dvc/) |

### 🛰️ 编排（Airflow）
五个 DAG，配备**企业级防护措施**：以执行日期为边界的幂等任务（可安全回填和重试）、`KubernetesPodOperator` 计算隔离、**reschedule 模式传感器**（不占用 worker 槽位）、并发池 + `max_active_runs=1`，以及指数退避重试。与编排无关的任务逻辑位于 [`src/pipelines/`](src/pipelines/)，并由 [`testing/`](testing/) 中的 DagBag 测试套件验证。

### 🚀 推理服务与部署
| 内容 | 位置 |
|---|---|
| FastAPI 服务与批量预测共享**完全相同**的代码路径（API 与 DAG 给出相同的预测） | [`serving/api.py`](src/serving/api.py) |
| API 密钥认证（常量时间比较）、**每分钟 60 次请求的限流**（防御模型窃取）、带哈希请求 ID 的审计日志、生产环境禁用 OpenAPI、输出异常防护 | [`serving/api.py`](src/serving/api.py) |
| 加固的 Kubernetes：非 root、`readOnlyRootFilesystem`、移除**全部** capabilities、seccomp `RuntimeDefault`、不挂载 SA token、**HPA 2→6 @ 70% CPU** | [`kubernetes/`](kubernetes/) |
| 强制执行 Pod 最小权限基线的 OPA/Rego 准入策略 | [`kubernetes/opa-policy.rego`](kubernetes/opa-policy.rego) |
| Locust 压测配置 | [`loadtest/`](loadtest/) |

### 🔐 安全与供应链
| 内容 | 位置 |
|---|---|
| **六阶段 CI/CD**：代码规范检查 → 代码评审 → 测试（+覆盖率门禁）→ 数据质量 → 冒烟训练 → 供应链 → 镜像构建 | [`.github/workflows/`](.github/workflows/mlops_pipeline.yaml) |
| 静态分析：**Bandit + Semgrep**（自定义架构规则）作为构建门禁 | [`.semgrep/rules.yml`](.semgrep/rules.yml) |
| **SBOM/AI-BOM**（CycloneDX）、依赖漏洞扫描（pip-audit）、**Trivy** 镜像扫描（发现 CRITICAL 即失败）、**cosign** 镜像签名 | CI 工作流 |
| **模型制品完整性**——加载前校验 SHA-256（防止被篡改的 pickle 在反序列化时执行恶意代码） | [`inference/model_integrity.py`](src/inference/model_integrity.py) |
| 每个批次都经过 PII 扫描与 pandera 数据质量门禁 | [`data/pii.py`](src/data/pii.py)、[`data/quality_gate.py`](src/data/quality_gate.py) |

### 📈 可观测性与运维
Prometheus + Grafana 仪表盘（请求速率、延迟直方图，以及通过 statsd-exporter 采集的漂移指标）、自动化的基础设施状态**备份/恢复**脚本，以及完整的交接运维手册。详见 [`observability/`](observability/)。

---

## 值得一提的工程决策

- **发现了数据泄漏**——否则会上线一个“看起来很好”的假模型；修复时也没有丢弃原有分析（[见上文](#为什么是生产级而不是一个-notebook)）。
- **风险是一等输出，而不是事后补充。** 运力规划需要的是一个*上界*，而不是点估计——因此系统提供经过实际覆盖率验证、并按小时调整波动率的 99% VaR 缓冲。
- **一个敢于说“不”的晋升门禁。** 没有门禁的持续训练是危险的：一次糟糕的重训练会悄无声息地拖垮线上效果。在这里，新模型必须*凭实力*赢得 `@champion`。
- **推理 API 与批量 DAG 运行同一套预测代码**——从结构上杜绝训练/推理偏差。
- **安全是“完成”的一部分**：最小权限 Pod、经过签名和扫描的镜像、制品完整性校验、限流，密钥只通过环境变量 / secret 引用传入。

---

## 快速开始

```bash
pip install -r requirements.txt
pip install -e .

# 获取公开 TLC 数据并构建原始事件文件（约 200 MB 下载）
python scripts/download_data.py --start 2024-01 --end 2024-04

# 端到端运行整条流水线（导入 → 训练 → 预测），无需 Airflow
python run_pipeline.py

# 复现样本外评估与图表
python scripts/evaluate.py

# 测试（109 个单元 + 集成测试）
pytest tests/ testing/

# 在真实的 Airflow UI 中查看 DAG（Docker）
docker compose -f docker-compose.airflow.yml up -d --build   # http://localhost:8088
```
> 仓库中已提交体积很小的聚合数据 `data/*_demand.csv`，因此开箱即可运行；体积较大的原始 Parquet 由下载脚本获取（已加入 git 忽略）。

---

## 仓库结构
| 路径 | 内容 |
|---|---|
| [`src/forecasting/`](src/forecasting/) | 无泄漏预测器、训练、推理引擎、VaR |
| [`src/inference/`](src/inference/) | 风险/VaR、输入校验、模型完整性防护 |
| [`src/pipelines/`](src/pipelines/) | 幂等的 DAG 任务逻辑（不依赖 Airflow，可完全测试） |
| [`src/orchestration/`](src/orchestration/) | Airflow 算子工厂（计算隔离、重试） |
| [`src/monitoring/`](src/monitoring/) | 漂移闭环与基于实际值的评分 |
| [`src/serving/`](src/serving/) | FastAPI 应用（认证、限流、指标） |
| [`src/data/`](src/data/)、[`src/features/`](src/features/)、[`src/governance/`](src/governance/) | 数据导入、Feast 特征、模型卡 |
| [`dags/`](dags/)、[`testing/`](testing/) | 轻量 Airflow DAG 与 DagBag 校验套件 |
| [`kubernetes/`](kubernetes/)、[`docker/`](docker/) | 加固的部署清单与镜像 |
| [`observability/`](observability/) | Prometheus + Grafana 监控栈 |
| [`tests/`](tests/) | 109 个单元/集成测试，含对抗性用例 |

---

## 技术栈
`Python 3.11 · LightGBM · GARCH/arch · Apache Airflow · MLflow · Feast · Evidently ·
FastAPI · Docker · Kubernetes · MinIO/S3 · Prometheus · Grafana · pandera · DVC ·
GitHub Actions · Bandit · Semgrep · Trivy · cosign · CycloneDX`

## 数据与许可证
行程数据：**纽约 TLC 开放数据**（公开）。镜像路径（`ghcr.io/your-username/…`）和基础设施地址均为占位符，请按你自己的环境设置。代码基于 **MIT** 许可证发布，详见 [LICENSE](LICENSE)。
