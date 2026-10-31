# PolyEvo

可插拔的聚合物 / 分子进化优化框架。需求见 [`docs/PRD.md`](docs/PRD.md)。

**当前完成的是 §12 第一步（M1 最小纵切），不是整份 PRD。** 预测值均为 Mock 合成数值，不具有物理意义，不能用来证明材料性能提升。

## 快速运行

需要 Python 3.11+。推荐在独立环境中安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

polyevo validate presets/mock_demo.yaml
polyevo list-ops --stage mutation
polyevo run --config presets/mock_demo.yaml --out runs/mock_demo
python -m pytest -q
```

也可用 `python -m polyevo` 代替 `polyevo`。输出目录必须不存在或为空，避免覆盖已有实验；目前不支持 `--resume`。

Demo 默认初代 24 个体，执行 20 次繁殖，输出 **Gen0–Gen20 共 21 个种群快照**。它使用固定归一化，因此精英保留下的 best fitness 单调不降。改成 `minmax_pool` 后，每代尺度变化，**跨代 fitness 不再可直接比较，也不承诺单调性**。

## 当前支持

| 部分 | M1 实现 |
|---|---|
| 核心 | Individual、只读配置 Context、注册表、阶段钩子、停止回调 |
| 域 | 线性聚合物两个 `*`；小分子零个 `*`；规范化、sanitize、连接点验证 |
| 编码 | `fragment_seq`（双端序列）+ `fragment_graph`（多端拓扑装配） |
| 建库 | polymer CSV → BRICS 全切分 → 过滤/去重 → 带端口元数据的 Parquet |
| 算子 | `brics_single_point`、`substitute`、`insert`、`delete`、`swap_order` |
| 流程 | `random_assemble` → `tournament` / `family` → `generational_elitist` |
| 评分 | maximize/minimize；minmax_pool、minmax_fixed、none；weighted_sum |
| 约束 | RDKit 描述符范围、禁用 SMARTS；有限次 resample |
| Oracle | Mock、批预测、SQLite 缓存、逐预测器调用账本、总预算 |
| 输出 | resolved YAML、片段 Parquet、代际 Parquet、summary CSV、archive、lineage、账本 |
| CLI | validate、run、list-ops、build-library |
| 预设 | `mock_demo.yaml`（演示）、`ga_mmpolymer.yaml`（论文对标，Mock 预测器） |

### 两条核心保证

1. **属性缓存，分数重算。** 缓存键是 `(predictor.name, canonical_smiles)`；每轮对父代与子代合并池重新评分。Archive 快照不保留分数；代际 Parquet 保留评分用于审计，但不作为评分缓存。
2. **先结构检查，后预测。** 初始化与每个算子出口都经过结构检查。不合法结构不会进入 oracle。当前池与最终子代去重；跨代再次出现的结构复用缓存。

预测异常、错误返回形状或非有限输出记为 NaN，并缓存失败；涉及失败目标的个体 fitness 为负无穷。日志中的 JSON 字段用 `null` 表示非有限属性。预算按**每个预测器评估一个结构**计一次（包括失败），不是按批次数计。若最后只够评估某个结构的部分属性，已经执行的调用照常记账，该结构不进入候选池。

`max_attempts` 限制初始化/繁殖重试次数。初始化凑不齐种群时明确报错；繁殖不足时用父代补位，无可用子代时以 `search_exhausted` 正常停止。`generational_elitist` 的 archive 保存本代存活者的精英，不是跨代历史最优集合。

### 片段表示的边界

M1 的片段库为 YAML 中的**双端、带 BRICS 同位素标签**片段，例如 `[16*]c1ccc([16*])cc1`。连接使用 RDKit 的 BRICS 兼容表；不兼容片段返回失败，不强行拼接。聚合物输出的两个剩余端口变为 `*`，小分子输出以 H 封端。

当前不是任意 BRICS 分解片段的通用装配器：不接收单端、多端或无标签片段；不支持片段方向基因、分支或自适应权重。`n_fragments` 仅控制**初始化**长度，进化中的尺寸可通过 MolWt 结构约束限制。演示库具有兼容端口；任意外部库不承诺同样的算子成功率。

## 从真实 polymer 建 BRICS 片段库

```bash
python -m polyevo build-library --input examples/polymers.csv --out runs/library.parquet

# 自定义列名与过滤；--forbid-smarts 可重复指定
python -m polyevo build-library --input polymers.csv --out runs/filtered.parquet \
  --smiles-column smiles --min-heavy-atoms 2 --max-heavy-atoms 60 \
  --allowed-elements C H N O S F Cl Br --forbid-smarts '[N+](=O)[O-]'
```

输入是含 `smiles` 列的 UTF-8 CSV（支持 BOM），每行一个含两个原始 `*` 的连通重复单元。带 BRICS 同位素标签的输入端点会被拒绝，避免误把已拆分片段当成原始 polymer。立体化学保留，原子映射号在规范化时去除。

采用 `FindBRICSBonds → BreakBRICSBonds → GetMolFrags`：**一次切断所有 BRICS 可断键**，不枚举重叠的中间片段。无可断键时保留整个重复单元；单端、多端、含原始聚合物端点的片段全部保留，不因当前装配器限制而丢弃。

例如 `*CCOC(=O)c1ccc(C(=O)OCC*)cc1` 拆成：

```text
*CC[4*]                  ×2  # * 是原始聚合物端点，[4*] 是 BRICS 端口
[1*]C([6*])=O            ×2
[3*]O[3*]                ×2
[16*]c1ccc([16*])cc1     ×1
```

Parquet 每行一个规范化后的唯一片段，含 `smiles`、默认 `weight=1.0`、`heavy_atoms`、`n_attachment`（原始端点数）、`n_brics_ports`、`brics_labels`、`occurrences`、`source_count`。重复 polymer 先去重，不重复贡献计数；过滤只作用于片段，重原子计数不含哑原子。元素白名单始终豁免两类 `*` 端口。

命令输出 JSON 报告：输入/非法/重复数、切键数、过滤数、最终规模和端口类型统计。非法行跳过并记录前 20 条行号与原因；没有保留任何片段则报错，不写空库。报告、过滤配置和 RDKit 版本也存入 Parquet schema metadata。已有目标文件绝不覆盖。

```python
from polyevo.encoding.library import decompose_polymer, build_library
import pyarrow.parquet as pq

parts = decompose_polymer('*CCOCC*')
print(parts.bonds, parts.fragments)  # 断键索引基于 parts.smiles 的规范化原子顺序
report = build_library('examples/polymers.csv', 'runs/python_library.parquet')
table = pq.read_table('runs/python_library.parquet')
```

**接入完成：`fragment_graph` 已支持从 Parquet 库直接加载并装配任意端口拓扑。** 原始链端（`*`）保留，BRICS 端口按兼容规则连接或转为新链端，最终恰好两个链端。单端、多端片段均可参与组装。

## 论文对标配置

`presets/ga_mmpolymer.yaml` 已配置：
- `fragment_graph` 编码 + `data/ga_mmpolymer/library.parquet`（2,450 个唯一片段）
- 三目标：Tg maximize / eps_r minimize / SA minimize，minmax_pool 归一化
- tournament(k=3) → family(20×6) → generational_elitist(0.10)
- Mock 预测器占位；替换为真实 MMPolymer 预测器后即可复现论文流程

运行：
```bash
python -m polyevo run --config presets/ga_mmpolymer.yaml --out runs/ga_mmpolymer
```

## Python API 与插件

```python
from polyevo import run_ga
from polyevo.config import load_config

state = run_ga(load_config("presets/mock_demo.yaml"), out="runs/python_demo")
print(state.stop_reason, state.oracle_calls)
```

注册选择器后，YAML 中即可引用：

```python
from polyevo import register

@register("select", "uniform")
def uniform(population, ctx):
    return population[int(ctx.rng.integers(len(population)))]

cfg = load_config("presets/mock_demo.yaml")
cfg["select"] = "uniform"
state = run_ga(cfg, out="runs/uniform")
```

CLI 插件可写作 `select: "python: my_package:uniform"`，模块须在 Python 导入路径上。**只运行可信配置**：`python:` 钩子会导入并执行 Python 代码。`validate` 同样会导入插件和实例化编码/预测器。

钩子签名以当前实现为准：`init(ctx, make_candidate)`、`score(population, ctx, aggregate)`、`select(population, ctx)`、`reproduce(population, ctx, select, make_candidate)`、`survive(candidates, parents, ctx) -> (population, archive)`、`diversity(population, ctx)`、`stop(ctx, population)`。编码的 genotype 可为任意对象，但当前落盘对非 JSON 类型只写 `repr`，**不能用于恢复**。

`run_ga(..., callbacks=[obj])` 支持对象上的 `on_start`、`on_generation_begin`、`on_oracle_end`、`on_generation_end`、`on_stop` 方法，签名 `(ctx, population)`；返回 `"stop"` 请求在安全阶段边界停止。`on_oracle_end` 收到本次完成评估的个体。Context 配置和编码映射只读；共享 RNG、预算和 archive 有意保留为运行状态。

## 输出

```text
runs/mock_demo/
  config.resolved.yaml
  fragment_library.parquet
  generations/gen_000.parquet ... gen_020.parquet
  summary.csv
  archive.parquet
  lineage.jsonl
  oracle_ledger.csv
  predictor_cache.sqlite
  run.json
```

Parquet 中 `genotype / props / scores / penalties` 是 JSON 字符串，其他 Individual 字段使用 Arrow 类型；lineage 记录成功完成评估的结构首次出现时的父代与操作。`run.json` 记录停止原因、代数和 oracle 次数。摘要目前含种群规模、属性及 fitness 统计、调用数，不含 Tanimoto 和 schedule 统计。

## 验证与下一步

测试覆盖：固定尺度下的 20 代运行；相对评分重新计算；跨实例 SQLite 缓存与失败缓存；非法结构不触发预测；部分预算；每个已实现算子在两个域中各 1,000 次随机应用；固定环境中同 seed 输出字节一致；插件、CLI 和配置校验。

按 PRD 后续推进：

- **M2**：真实预测器/Subprocess、六种 StarMode、属性约束、调度、完整 BRICS 库接入装配、checkpoint 恢复（建库 CLI 已完成）。
- **M3**：graph 编码、Pareto/NSGA-II、PMO 接入。
- **M4**：其他生存策略、修饰链、多样性与分析报告。

尚未实现的字段、注册名和 CLI 参数会报错，不会当作已支持功能吞掉。跨 RDKit/NumPy 版本不保证逐位一致；更换预测器实现或模型版本必须换唯一的 property/name，不能复用旧缓存身份。
