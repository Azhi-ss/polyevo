# PRD：PolyEvo — 可插拔的聚合物 / 分子进化优化框架（核心逻辑 v2）

> 给实现 Agent：这是一个**通用框架**，论文 GA-MMPolymer 的 GA 只是预设 `presets/ga_mmpolymer.yaml`。先做 §12 的最小纵切，再铺零件表。文中 **[论文值]** = 默认值来自论文 SI Table S7/S5。
> 设计哲学：**引擎只认阶段与数据契约，不认具体算法**。每个阶段的行为由注册表中的可调用对象决定；YAML 只是拼注册名。

---

## 1. 目标与边界

输入：搜索空间（编码 + 片段库）、任意多个属性预测器、目标/约束、算法配置。
输出：每代种群与属性、谱系、oracle 账本、档案（最优/Pareto/格子）、分析图。
域：线性聚合物重复单元（两个 `*`）与小分子，一套代码。

不做：训模型；DFT/MD；共聚序列、支链/交联；Ray/Dask；复制 PMO 任务集。

---

## 2. 两条铁律（贯穿所有设计）

**铁律 1：属性是绝对的，分数是相对的。**
- `props`（预测器输出）只取决于 SMILES，**按 SMILES 永久缓存**，一个结构一生只查一次 oracle。
- `scores / fitness / rank` 取决于"和谁比"（归一化范围、Pareto 排序、niche），**每代对候选池整体重算**，不存档、不跨代复用。
→ 这样 μ+λ、精英回流、档案个体再评分都不会出现"父代用旧尺子、子代用新尺子"的问题。

**铁律 2：便宜的检查永远在昂贵的检查之前。**
- 结构合法性与 RDKit 可算约束（分子量、SMARTS、连接点）在**算子产出瞬间**执行，不合格直接不进候选池、不耗 oracle。
- 需要预测值的约束只能在 oracle 之后处理。

---

## 3. 核心对象与数据契约

```python
Genotype     # 任意对象，由 Encoding 定义；必须可 decode 为 SMILES
Individual
  id: str            # = 规范化 SMILES（去立体可配）；同 SMILES 即同个体，不论编码
  encoding: str
  genotype: Genotype
  smiles: str
  props: dict[str, float]        # 绝对，缓存；NaN = 该预测器失败
  scores: dict[str, float]       # 相对，每代重算；归一化后统一"越大越好"
  penalties: dict[str, float]    # 相对或绝对均可，∈ (0,1]
  violation: float               # 约束违反量总和，0 = 可行（constraint_dominance 用）
  fitness: float | None          # 标量；pareto 模式下 = None
  rank: int | None; crowding: float | None
  parents: tuple[str, ...]; ops: list[str]; gen_born: int; age: int

Context      # 传给每个钩子的只读环境
  cfg, rng, generation, progress ∈ [0,1], spec: ConnectivitySpec,
  encodings: dict[str, Encoding], library, budget: BudgetLedger, archive: Archive

Archive      # 由 survive 策略拥有并定义形态：
             # best_k | pareto_front | map_elites_grid | none
```

`progress = max(generation / max_generations, oracle_calls / budget)`，调度函数可按 `generation` 或 `progress` 定义，后者让"固定代数"和"固定预算"两种跑法共用同一套调度。

---

## 4. 域：`ConnectivitySpec`

```yaml
domain: {name: polymer_linear, n_attachment: 2, attachment_symbol: "*", forbid_adjacent: true}
# molecule → n_attachment: 0，所有 * 相关逻辑 no-op
```
所有算子必须守恒 `n_attachment`。守恒由框架在算子外层统一校验（§8 阶段 ④），算子作者不必自己写。

---

## 5. 编码：`Encoding`

```python
class Encoding(Protocol):
    name: str
    def random(self, ctx) -> Genotype
    def decode(self, g) -> str | None
    def encode(self, smiles) -> Genotype | None        # 可选
    crossover_ops: dict[str, Callable[[Genotype, Genotype, Context], Genotype]]
    mutation_ops:  dict[str, Callable[[Genotype, Context], Genotype]]
    def distance(self, a, b) -> float                   # 默认 1 − Tanimoto(decode)
```

| 编码 | v1 | 原生算子 |
|---|---|---|
| `fragment_seq`（BRICS 片段序列） | ✅ | 交叉 `brics_single_point / brics_two_point / fragment_swap`；变异 `substitute / insert / delete / swap_order / crem_substitute`(可选依赖 CReM) |
| `graph`（RDKit Mol，Graph GA） | ✅ | 交叉 `graph_crossover`；变异 `atom_insert/delete/change, bond_change, ring_append/remove` |
| `string`、`synthesis` | 接口 | v2；v1 只保证 `genotype` 可装任意对象并走 `decode` |

混合编码：一个 run 可同时启用多种；交叉只在同编码父代间进行；`reproduce` 按配置比例分配各编码的子代名额。

### 5.1 片段库
BRICS 分解保留哑原子标签；过滤（去重、重原子范围、禁用 SMARTS、元素白名单）；每片段带 `weight`，可静态或 `adaptive`（按上一代存活者中出现频率更新）；parquet 存取；报告实际规模（论文 ≈ 4,000，源 12,533 [论文值]）。

---

## 6. 预测器：`Predictor`

```python
class Predictor(Protocol):
    name: str; unit: str | None
    star_mode: StarMode           # keep | cap_H | cap_CH3 | cap_custom:<smi> | oligomer:<n> | wildcard_atom:<sym>
    batch_size: int | None
    def predict(self, smiles: list[str]) -> np.ndarray            # NaN = 失败
    def predict_with_uncertainty(self, smiles) -> (mean, std)     # 可选
```

内置：`SAScore`, `RDKitDescriptor(name)`, `Callable(fn)`, `Subprocess(cmd)`, `HTTP(url)`, `Ensemble([...])`(自动给 std), `Mock`, `PMOOracleAdapter`。
所有预测器自动套 `Cached`：键 = (predictor.name, 转换后 SMILES)，sqlite 持久化；**只有缓存未命中才计入 oracle 账本**。
`*` 转换由框架按 `star_mode` 统一做；失败 → NaN；`molecule` 域下 no-op。
账本按预测器分别记录调用序号，供 top-k AUC 复算。

---

## 7. 目标、约束、评分

### 7.1 目标
```yaml
objectives:
  - {property: Tg,    direction: maximize, weight: 0.50, normalize: minmax_pool}   # [论文值]
  - {property: eps_r, direction: minimize, weight: 0.35, normalize: minmax_pool}   # [论文值]
  - {property: SA,    direction: minimize, weight: 0.15, normalize: minmax_pool}   # [论文值]
```
`direction: maximize | minimize | target(v | [lo,hi])`
`normalize: minmax_pool`(当前候选池) `| minmax_gen0 | minmax_fixed:[lo,hi] | zscore_pool | rank | sigmoid(c,s) | none`

### 7.2 约束：分两层，按铁律 2 分开执行

**结构层**（oracle 前，阶段 ④）——只允许 RDKit 可直接算的量：
```yaml
structural_constraints:
  - {property: MolWt, range: [60, 1000], on_fail: drop}
  - {smarts_forbidden: ["[N+](=O)[O-]"], on_fail: resample}     # resample = 让算子重试 k 次
```
内置常开：RDKit 可解析/sanitize；连接点数 = `n_attachment`；连接点不相邻、一价。

**属性层**（oracle 后，阶段 ⑥）：
```yaml
property_constraints:
  - {property: eps_r, max: {percentile_gen0: 80}, mode: penalty,
     penalty: {type: exp_over_threshold, lambda: 0.10}}           # [论文值]
  - {property: logP, max: 5, mode: dominance}                      # Deb 规则，写入 violation
  - {property: Tg,   min: 100, mode: drop}                         # 评完直接淘汰（已耗 oracle，慎用）
```
`mode: penalty | dominance | drop`。`repair` 不作为通用模式——真正的修复是编码特定的，留给编码自己的 mutation 实现。

### 7.3 聚合
`weighted_sum`[论文] `| weighted_product | chebyshev | lexicographic(order) | pareto`
`pareto` 不产生标量 fitness，产生 `rank + crowding`。

### 7.4 修饰链（可空，按序应用于聚合结果）
`uncertainty_bonus(kappa)` / `novelty_bonus(k, w)`（对 archive 的 k 近邻距离）/ `age_penalty(rate)` / `fitness_sharing(sigma)`（相似个体互相压分——放在这里而不是单独阶段，因为它本质是改分数）/ `custom`。

最终：`fitness = modifiers(aggregate(scores) × Π penalties)`；所有中间量落日志。

### 7.5 兼容性规则（配置校验时强制）
| aggregation | 允许的 select | 允许的 survive |
|---|---|---|
| 标量类 | tournament / roulette / rank / sus / boltzmann / random | generational_elitist / mu_plus_lambda / mu_comma_lambda / map_elites |
| pareto | nsga2_binary_tournament / random | nsga2 |
含 `dominance` 约束时，所有比较先比 `violation`（可行优先），再比 fitness 或 rank。`polyevo validate` 对不兼容组合报错。

---

## 8. 引擎：八个阶段，一条流水线

```
① init ─▶ ② score(P₀) ─▶ ③ log/callbacks/stop? ─▶ ④ select+reproduce+structural_filter
   ─▶ ⑤ oracle(仅新 SMILES) ─▶ ⑥ score(候选池 = 存活者 ∪ 子代) ─▶ ⑦ survive ─▶ ⑧ diversity ─▶ 回到 ③
```

| # | 阶段 | 输入 → 输出 | 要点 |
|---|---|---|---|
| ① | `init` | ctx → Population | `random_assemble(n_fragments=[2,4])` / `from_file` / `mixed` / `from_archive`；产出经阶段 ④ 同款结构过滤，再过 ⑤ oracle |
| ② | `score` | 候选池 → 候选池（填 scores/penalties/violation/fitness/rank） | 唯一实现；**整池一起算**（铁律 1）。`percentile_gen0` 阈值在首次调用时定格 |
| ③ | `callbacks` + `stop` | ctx → bool | stop 多个取 OR：`max_generations(50)`[论文值] / `budget_exhausted` / `patience(n, metric)` / `target_reached` / `wallclock` |
| ④ | `select` → `reproduce` → 结构过滤 | Population → 子代列表（只有 genotype+smiles，无 props） | `select`: `tournament(k=3)`[论文值] / roulette / rank / sus / boltzmann / nsga2_binary_tournament / random。`reproduce`: `family(20×6)`[论文值] / `pairwise(n)` / `mutation_only(n)`；内部调编码的 crossover（prob，默认 1.0 [论文值]）与各 mutation（各自 schedule）。框架在每个算子出口做：decode → sanitize → 连接点守恒 → 结构约束 → 去重（对当前池 + 子代列表）；失败按 `on_fail` drop 或 resample |
| ⑤ | `oracle` | 子代中缓存未命中的 SMILES → props | 去重 → 按 star_mode 转换 → 分批 → 线程池 → 写缓存 → 记账本。**预算只够一部分时**：按子代顺序评估到预算用尽，其余子代丢弃，并置 `budget_exhausted` |
| ⑥ | `score` | 存活者 ∪ 子代 → 全部重算 | 同 ② |
| ⑦ | `survive` | 候选池 → (Population, Archive) | `generational_elitist(ratio=0.10, mode=replace\|append)`[论文值] / `mu_plus_lambda(mu)` / `mu_comma_lambda(mu)` / `nsga2(pop_size)` / `map_elites(descriptors, bins)`。**Archive 由 survive 拥有**：elitist → best_k，nsga2 → pareto_front，map_elites → grid |
| ⑧ | `diversity` | Population → Population | `none` / `niche_cap(threshold, max_per_niche)`；只做"删"，不改分数（改分数的 fitness_sharing 在 §7.4） |

变异 schedule（阶段 ④ 内）：`constant(p)` / `linear(p0, p1, g0, g1)`[论文值 0.25→0.08, 5→45] / `exponential` / `adaptive_1_5(target_rate)` / 外加 `sigma_offset(σ)`[论文值 0.15]：`p ← clip(p + N(0, σ·p), 0, 1)`。全部可按 `generation` 或 `progress` 定义。

### 8.1 回调
`on_start / on_generation_begin / on_oracle_end / on_generation_end / on_stop`，签名 `(ctx, population) -> None | "stop"`。内置：`CSVLogger`, `ParquetCheckpoint`, `ProgressBar`, `EarlyStopOnNaNRate`, 可选 `WandB/TensorBoard`。

### 8.2 可复现与恢复
单一 `rng` 经 `ctx` 传入所有钩子；RDKit 随机操作用从 `rng` 派生的 `randomSeed`。同 seed 两次逐位一致。`--resume` 从任意代 checkpoint 恢复，oracle 缓存持久化，恢复后结果与不中断一致。

---

## 9. 配置与 CLI

YAML 字段 = 注册名 | `{name, 参数}` | `python: module:callable`。

```
polyevo build-library --input polymers.csv --out lib.parquet
polyevo validate cfg.yaml            # 解析全部钩子 + §7.5 兼容性检查 + 打印默认值
polyevo run --config cfg.yaml --out run_dir [--resume]
polyevo report run_dir [--generations 0,9,19,29,39,49]
polyevo predict --config cfg.yaml --smiles ...
polyevo list-ops [--stage mutation]
```
Python API：`run_ga(encodings=..., library=..., predictors=[...], objectives=[...], select=..., survive=..., callbacks=[...], seed=42) -> RunState`。

## 10. 输出
```
run_dir/ config.resolved.yaml  fragment_library.parquet  generations/gen_XXX.parquet(每个体一行，含 §3 全部字段)
         summary.csv(每代：pop_size, 各属性 mean/best/std, fitness 统计, mean/std Tanimoto, 各 schedule 当前值, 累计 oracle)
         archive.parquet  lineage.jsonl  oracle_ledger.csv(含调用序号)  predictor_cache.sqlite  figures/
```

## 11. 分析（解耦，对任意 run_dir 事后运行）
属性曲线；fitness 分布；Tanimoto（Morgan r=2, 2048, 手性 [论文值]）逐代 mean±std 与指定代热图；t-SNE/UMAP；任意两属性散点 + 目标窗 + 第三属性着色；Pareto 前沿演化；MAP-Elites 格子热图；顶尖子集 motif 富集与描述符对比；**top-k AUC**（PMO 口径，按账本序号复算）；`polyevo report` 一键 PNG + markdown。

## 12. 实施顺序（最小纵切优先）
1. `core`（Individual/Context/registry/engine 八阶段）+ `fragment_seq` + `tournament` + `family` + `generational_elitist` + `Mock` 预测器 + `weighted_sum` → `presets/mock_demo.yaml` 跑通
2. `Cached`/账本/`Subprocess` 预测器 + 全部 schedule + 结构/属性约束 → `presets/ga_mmpolymer.yaml`
3. `graph` 编码 + `nsga2` + `pareto` 聚合 + `PMOOracleAdapter` → `presets/molecule_graph_ga_nsga2.yaml`
4. `mu_plus_lambda` / `map_elites` / 修饰链 / `niche_cap` / 分析模块 → `presets/map_elites_polymer.yaml`

## 13. 验收
**单元**
- 两个编码 × 全部算子：1,000 次随机应用，连接点守恒、可 sanitize、非法率 < 30%
- StarMode 六模式结果固定；`oligomer:3` 重原子数正确
- `linear` schedule g=0,5,25,45,49 → 0.25, 0.25, 0.165, 0.08, 0.08；`progress` 版在 50 代下等价
- 评分：手工 props → 手算 fitness；7 归一化 × 5 聚合各一例；penalty 阈值下 = 1；dominance 排序正确
- **铁律 1 测试**：同一 SMILES 在不同代 props 相同且 oracle 只调一次；精英个体每代 fitness 随池变化而变
- **铁律 2 测试**：违反结构约束的子代不产生任何 oracle 调用
- 五种 survive 下一代规模与 archive 形态正确；nsga2 rank/crowding 与参考实现一致
- `niche_cap` 后任意两个体距离 ≥ threshold 或同 niche 内 ≤ max
- 预算中途耗尽：评估数 = 剩余预算，run 正常停止并落盘
- 同 seed 逐位一致；`--resume` 与不中断一致
- `validate` 对 `pareto + tournament` 等不兼容组合报错；用户 `@register` 后可从 YAML 引用

**集成**
- 四预设跑通；`mock_demo` 20 代 best fitness 单调不降
- `ga_mmpolymer` 接真 MMPolymer：趋势对照论文 Table S8（mean Tg ↑、mean εr ↓、mean Tanimoto ↑ 但 < 0.5、Gen49 目标窗 Tg ≥ 260 ℃ & εr ≤ 2.8 命中率显著高于 Gen0）。数值不要求一致；**禁止**为凑数调预测器或挑种子
- `molecule_graph_ga_nsga2` 在任一 PMO 任务上 top-10 AUC ≥ mol_ga 默认的 80%（同预算 10k）

## 14. 包结构
```
polyevo/ core/(individual context registry engine) domain/(connectivity starmode validity)
           encoding/(base fragment_seq/ graph/) predictors/(base builtin cache budget)
           fitness/(objectives constraints aggregate modifiers) operators/(select reproduce schedule survive diversity stop)
           callbacks/ analysis/ cli/ presets/   tests/
```
