# PharmFlow

**PharmFlow（药房流程）**：基于 Isaac Lab–Arena 的**药品到传送带**（biomedical_droid）数据采集项目，专注「从打开的纸箱中抓取药瓶 → 旋转扶正 → 放到传送带上」这一数据采集链路。

**PharmFlow 是一个轻量、自包含的数据采集项目**，聚焦于 biomedical 数据采集链路，提供可复现的最小代码与资产。

---

## 特性

- **DROID 单机械臂拣放**：使用 Arena 的 DROID 关节空间/差分 IK embodiment
- **双控制器**：
  - `--controller auto`：基于 IsaacLab Mimic cuRobo 规划器的脚本化专家（自动收集）
  - `--controller keyboard`：IsaacLab 键盘遥操作
- **最小化场景**：只含工作台、传送带（含辊筒）、打开的药箱、10 个药瓶——无门诊背景、无分拣机
- **HDF5 数据导出**：Episode 记录为 HDF5，动作遵循 DROID 8 维契约（7 个手臂关节目标 + 1 个夹爪开合）
- **全自包含第三方管理**：`isaaclab_arena` / `isaaclab_arena_curobo` vendored 进仓库；`isaaclab` 本体由 IsaacLab 子模块 + 定制补丁提供

---

## 安装

### 前置条件

- **操作系统**：Linux（主平台），必须 NVIDIA GPU
- **Python**：3.12
- **CUDA**：12.8（推荐）
- **NVIDIA 驱动**：570.x（推荐）
- **硬件**：推荐 NVIDIA RTX GPU

### 1. 克隆仓库（含子模块）

```bash
git clone --recurse-submodules https://github.com/zmf2022/PharmFlow.git
cd PharmFlow
```

> `--recurse-submodules` 会拉取 `third_party/IsaacLab` 子模块（`release/3.0.0-beta2`）。若已克隆但未带子模块，执行：
> ```bash
> git submodule update --init third_party/IsaacLab
> ```

### 2. 创建并激活环境

```bash
conda create -n pharm_flow python=3.12 -y
conda activate pharm_flow
export PHARM_FLOW_ROOT=$PWD   # 建议写入 ~/.bashrc 或 shell 配置
```

### 3. 执行安装脚本

```bash
bash scripts/install.sh
```

`install.sh` 依次完成：
1. 初始化 `third_party/IsaacLab` 子模块
2. 将 **cuRobo planner 定制补丁**应用到 `isaaclab_mimic`（见下文「定制补丁」）
3. 安装 Isaac Sim `6.0.1` + IsaacLab 全部源码扩展（含 `isaaclab_mimic`）
4. 以 editable 安装 PharmFlow
5. 安装 cuRobo `v0.7.8`（v1 API，`curobo.cuda_robot_model` 布局）
6. 应用 OpenBLAS 单线程修复（避免启动 SIGSEGV）

### 4. 拉取 LFS 资产

PharmFlow 的二进制资产（药瓶、纹理）用 Git LFS 管理：

```bash
git lfs pull
```

> 直接 clone 不自动拉取 LFS 内容，若场景加载失败请先执行上述命令。

---

## 运行

### 自动数据采集（无需人工，脚本化专家 + cuRobo）

```bash
python pharm_flow/data_collection/collect.py \
    --task biomedical_droid \
    --controller auto \
    --visualizer none \
    --num-demos 50
```

收集 **50 条成功示范** 到 `pharm_flow_logs/datasets/biomedical_droid/`。

### 键盘遥操作采集（需 GUI）

```bash
python pharm_flow/data_collection/collect.py \
    --task biomedical_droid \
    --visualizer kit \
    --dataset-dir pharm_flow_logs/datasets/biomedical_droid \
    --controller keyboard \
    --num-demos 20
```

> 键盘模式不能用 `--headless`（需要 IsaacLab GUI）。

### 常用参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--task` | 采集任务名 | `biomedical_droid` |
| `--controller` | `auto`（cuRobo 专家）/ `keyboard` | `keyboard` |
| `--visualizer` | `kit`（GUI）/ `none`（headless） | 自动 |
| `--num-demos` | 采集成功示范数量（0=不停止） | `0` |
| `--dataset-dir` | 输出目录 | `pharm_flow_logs/datasets/biomedical_droid` |
| `--seed` | 随机种子（缺省则每次随机药数） | `None` |
| `--scene-config` | 场景 YAML | `pharm_flow/config/scenes/biomedical.yaml` |
| `--disable-background` | 不加载背景（现无背景，保留兼容） | 关闭 |

---

## 项目结构

```
PharmFlow/
├── pharm_flow/                     # 业务 Python 包
│   ├── config/scenes/biomedical.yaml   # 场景定义（工作台/传送带/药瓶）
│   ├── data_collection/
│   │   ├── collect.py              # 采集入口
│   │   ├── tasks/                  # 任务适配（BiomedicalDroidTask）
│   │   ├── arena/                  # Arena 环境/任务/embodiment 适配
│   │   ├── utils/                  # 运行器、录制、技能、策略
│   │   └── experts/                # cuRobo 专家（medicine_pick_place）
│   ├── envs/scene_motion.py        # 传送带等场景运动控制
│   └── utils/monkey_patch.py       # IsaacLab 运行时补丁
├── third_party/
│   ├── IsaacLab/                   # git 子模块 (release/3.0.0-beta2)
│   ├── isaaclab_arena/             # vendored 顶层包（DROID embodiment 等）
│   └── isaaclab_arena_curobo/      # vendored 顶层包（cuRobo 环境适配）
├── assets/                         # 药瓶 + 药箱（Git LFS）
├── scripts/
│   ├── install.sh                  # 环境安装
│   └── patches/isaaclab-curobo-planner.patch   # cuRobo 定制补丁
├── pyproject.toml                  # 项目元数据与依赖
└── .gitmodules                     # IsaacLab 子模块声明
```

---

## 场景说明

`pharm_flow/config/scenes/biomedical.yaml` 描述核心工作台：

| 组件 | 类型 | 说明 |
|------|------|------|
| `workcell_table` | 图元 Cube | 桌面承载区 |
| `conveyor_base` / `conveyor_surface` | 图元 Cube | 传送带主体与皮带面 |
| `conveyor_roller_left/right` | 图元 Cylinder | 传送带辊筒（视觉） |
| `medicine_carton_open` | USD | 打开的医药纸箱（碰撞支撑） |
| `medicine_bottle_00..09` | USD 刚体 | 10 个药瓶（动态池，随机摆放） |

已移除：`outpatient_clinic`（门诊背景）、`sorting_machine`（分拣机）。工作台、传送带均由 YAML 图元生成，无独立资产文件。

**任务契约**：成功 = 药瓶被夹爪抓起 → 旋转扶正（长轴与世界 Z 对齐，`upright_axis=[0,1,0]`）→ 放置在传送带有效区域内 → 释放。成功以**物体位姿 + 夹爪释放**判定（非关节到位）。

---

## 依赖管理说明

### 第三方包来源

| 包 | 来源 | 说明 |
|----|------|------|
| `isaaclab` 本体 | `third_party/IsaacLab` 子模块 → `isaaclab.sh --install` | 框架，Python 3.12 + Isaac Sim 6 |
| `isaaclab_mimic` | 同上（IsaacLab source 扩展） | cuRobo 运动规划（apply 补丁后） |
| `isaaclab_arena` | vendored `third_party/isaaclab_arena` | 场景/任务/embodiment 组装 |
| `isaaclab_arena_curobo` | vendored `third_party/isaaclab_arena_curobo` | cuRobo 环境适配 |
| `isaacsim` | PyPI/NVIDIA index `6.0.1.0` | 模拟器运行库 |
| `curobo` | GitHub `v0.7.8` | cuRobo V1（运动规划底层） |

### vendored 包的导入

`isaaclab_arena` / `isaaclab_arena_curobo` 保持**顶层包名**，`import pharm_flow` 时由 `pharm_flow/__init__.py` 把 `third_party/` 注入 `sys.path`，因此 `collect.py` 顶部的 `import pharm_flow` 即完成路径引导。它们无需单独 `pip install`。

### 定制补丁

`scripts/patches/isaaclab-curobo-planner.patch` 为官方 `isaaclab_mimic` 的 cuRobo 规划器添加 `use_cuda_graph` / `warmup` 两个配置项，把原本硬编码的行为改为**配置驱动**：
- 构造时是否进行 cuRobo 预热由 `warmup` 控制（原来总是 `enable_graph=True` 预热）
- 求解器是否用 CUDA graph 由 `use_cuda_graph` 控制

这样 `biomedical.yaml` 的 `auto_collection` 里 `warmup: false` 与 `use_cuda_graph: false` 才能真正生效，避免 `PRIMITIVE` 碰撞检查器被强制开启 cuda graph。install 时自动 apply；重复 apply 会因 `git apply --check` 失败而跳过（幂等）。

---

## 数据格式

导出 HDF5 动作遵循 **DROID 8 维契约**：
- 前 7 维：DROID 手臂关节位置目标
- 第 8 维：二进制夹爪命令（`0`=开，`1`=合）

采集时使用 cuRobo 生成绝对关节轨迹（`--controller auto`）或 Arena 差分 IK + 键盘遥操作（`--controller keyboard`）。

---

## 常见问题

- **启动段错误 (exit 139)**：通常为 scipy OpenBLAS 与 Isaac Sim fork 冲突。`install.sh` 已写入 `OPENBLAS_NUM_THREADS=1` 并持久化到 conda activate 钩子；若仍出现，手动 `export OPENBLAS_NUM_THREADS=1`。
- **场景加载失败**：请确认执行过 `git lfs pull`（药瓶/纹理在 LFS）。
- **`import isaaclab_arena` 未解析**：确认 `PHARM_FLOW_ROOT` 或从项目根运行 `collect.py`，使 `pharm_flow/__init__.py` 注入 vendored 路径。
- **cuRobo 报 `collision_checker_type` 配置项缺失**：说明定制补丁未应用；重新运行 `git apply scripts/patches/isaaclab-curobo-planner.patch`（在 `third_party/IsaacLab` 内）。

---

## License

本项目为内部/研究用途示例，核心流程构建于 NVIDIA Isaac Lab–Arena（Apache 2.0 等开源许可）。请遵循各第三方依赖的许可条款。
