#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# PharmFlow 环境安装脚本
#
# 前置条件（从头开始配置）:
#   conda create -n pharm_flow python=3.12
#   conda activate pharm_flow
#   export PHARM_FLOW_ROOT=/path/to/PharmFlow
#   bash scripts/install.sh
#
# 结构:
#   third_party/IsaacLab            git 子模块 (release/3.0.0-beta2)
#   third_party/isaaclab_arena      vendored 顶层包
#   third_party/isaaclab_arena_curobo  vendored 顶层包
#
# install 动作:
#   1. 初始化 IsaacLab 子模块
#   2. 应用定制 cuRobo planner 补丁到 isaaclab_mimic
#   3. 安装 Isaac Sim 6.0.1 (+ isaaclab 全部源码扩展, 含 mimic)
#   4. cuRobo v0.7.8 (v1 API)
#   5. OpenBLAS 单线程修复 (避免 startup SIGSEGV)
#
# 注意: vendored 的 isaaclab_arena / isaaclab_arena_curobo 借助
#       pharm_flow/__init__.py 的 sys.path 注入被导入，无需单独 pip 安装。
# ============================================================

# Isaac Sim 6 / Isaac Lab 3 要求 Python 3.12.
python -c 'import sys; assert sys.version_info[:2] == (3, 12), "PharmFlow with Isaac Sim 6.0.1 requires Python 3.12"'

PROJECT_ROOT="${PHARM_FLOW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
echo "Installing PharmFlow at: $PROJECT_ROOT"

python -m pip install --upgrade pip

# ── 1. 初始化 IsaacLab 子模块 ───────────────────────────────────────────────
git submodule update --init third_party/IsaacLab

# ── 2. 应用定制 cuRobo planner 补丁到 isaaclab_mimic ───────────────────────
# 该补丁为 isaaclab_mimic 的 CuroboPlannerCfg 增加 use_cuda_graph / warmup
# 可配置项，biomedical collection 的 expert 依赖它们。
pushd third_party/IsaacLab
if git apply --check ../scripts/patches/isaaclab-curobo-planner.patch; then
    git apply ../scripts/patches/isaaclab-curobo-planner.patch
    echo "Applied isaaclab-curobo-planner.patch"
else
    echo "WARNING: 补丁未能应用（可能已应用或源码版本不同）。"
fi
popd

# ── 3. 安装 Isaac Sim 6.0.1 运行库 + isaaclab 全部源码扩展(含 mimic) ────────
python -m pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com

# isaaclab.sh 会自动使用当前 conda 环境的 python。
cd third_party/IsaacLab
bash isaaclab.sh --install

# ── 4. 安装 PharmFlow 本体（editable）─────────────────────────────────────────
cd ../..
python -m pip install -e . --no-deps

# ── 5. 安装 cuRobo v1 API（curobo.cuda_robot_model 布局）────────────────────
# v0.7.8 是官方 v1 最新版；v0.8.0 是破坏性 cuRoboV2 重写。
python -m pip uninstall -y nvidia-curobo curobo || true
python -m pip install \
    "git+https://github.com/NVlabs/curobo.git@v0.7.8" \
    --no-build-isolation

# ============================================================
# OpenBLAS 兼容性修复：
# scipy 的 OpenBLAS 线程池与 Isaac Sim 的 fork 冲突会导致
# 启动段错误 (SIGSEGV, exit 139)。限制 OpenBLAS 单线程可避免。
# ============================================================
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    OPENBLAS_SH="$CONDA_PREFIX/etc/conda/activate.d/openblas.sh"
    mkdir -p "$(dirname "$OPENBLAS_SH")"
    if ! grep -q OPENBLAS_NUM_THREADS "$OPENBLAS_SH" 2>/dev/null; then
        echo 'export OPENBLAS_NUM_THREADS=1' >> "$OPENBLAS_SH"
        echo 'export CARB_CRASH_REPORTING=0' >> "$OPENBLAS_SH"
    fi
fi
export OPENBLAS_NUM_THREADS=1
echo "OpenBLAS single-thread fix applied"

echo "Installation complete. Run PharmFlow with:"
echo "  export PHARM_FLOW_ROOT=$PROJECT_ROOT"
echo "  python pharm_flow/data_collection/collect.py --task biomedical_droid --controller auto --visualizer none --num-demos 50"
