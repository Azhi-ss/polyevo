# 补充数据（来源：github.com/LZ0221/Lieconv-Tg）

2026-10-29 从该仓库 `polymer screening/` 提取并单位对齐。**未并入 `../Tg.csv`**，需要时自行拼接。

| 文件 | 行数 | 列 | 来源与性质 |
|---|---|---|---|
| `polyaskinG_Tg.csv` | 56,762 | `psmiles, Tg_C` | PolyAskInG 数据库 Tg，原单位 **K**，已换算 ℃。剔除 3 条原值超 100–850 K、54 条与 `../Tg.csv` 重复。聚酰亚胺为主，Tg 120.8–564.8 ℃，中位 269.8 |
| `PI1M_MF3_structures.csv` | 440,232 | `psmiles, Tg_pred_C, SA_score` | PI1M 虚拟库的 **模型预测值**（非实验标签），°C。仅可作结构库/初始化池，**不能当训练标签** |

## 该仓库没有的
Lieconv-Tg 论文所述 **7,166 条均聚物训练集不在仓库里**——`Dataset generation and Feature representation/` 只有代码，读的是本地 `data.csv` / `data.sdf`。README 也未给外部下载链接。所以本仓库无法提供那批实验 Tg。

## 单位陷阱
同一仓库里三种文件单位不同：
- `PI1M_MF_*` 的 `Tg_pred` → **℃**（中位 67，min −1131，max 2566）
- `PolyAskInG_*` 的 `Tg_1` / `Tg_pred` → **K**（Tg_1 中位 543）

## 分布警告
PolyAskInG 这批全是大 Tg 聚酰亚胺/芳杂环（中位 270 ℃），和 `../Tg.csv`（中位 135 ℃，含大量聚丙烯酸酯/聚烯烃）分布差异极大。直接合训会有明显分布偏移；若只想扩结构多样性而不改标签分布，建议按 Tg 分箱后采样合并。
