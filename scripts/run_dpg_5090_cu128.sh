#!/usr/bin/env bash
# Build a separate Python 3.10 / cu128 DPG environment, then call the frozen scorer.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "${1:-}" in
    setup|smoke|setup-smoke) [[ $# == 1 ]] || exit 2 ;;
    dpg-group) [[ $# == 3 ]] || exit 2 ;;
    *) echo 'Usage: run_dpg_5090_cu128.sh setup | smoke | setup-smoke | dpg-group GROUP TOTAL' >&2; exit 2 ;;
esac
afs=/mnt/afs/L202500030/xietianyu
prefix="$afs/envs/scd-eval-dpg-5090-cu128-v1"
py="$prefix/bin/python"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
source "$afs/miniconda3/etc/profile.d/conda.sh"
if [[ "$1" == setup || "$1" == setup-smoke ]]; then
    exec 9>"$prefix.setup.lock"
    flock -n 9 || { echo 'Another job is installing this shared environment' >&2; exit 1; }
    if [[ ! -x "$py" ]]; then
        conda create -y --override-channels -c https://conda.anaconda.org/conda-forge --no-default-packages -p "$prefix" python=3.10 pip
    fi
fi
conda activate "$prefix"
export LD_LIBRARY_PATH="$prefix/lib/python3.10/site-packages/torch/lib:$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SCD_DPG_PY="$py" SCD_EVAL_PY="$py"
install_environment() {
    pins="$prefix/dpg-torch-constraints.txt"
    printf '%s\n' 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' 'torchaudio==2.7.1+cu128' > "$pins"
    "$py" -m pip install 'pip==24.0' 'setuptools==68.2.2' 'wheel==0.41.3' 'Cython==0.29.37' 'numpy==1.26.4'
    "$py" -m pip install --index-url https://download.pytorch.org/whl/cu128 -r "$pins"

    # OFA imports fairseq. Build its Cython extensions with no torch in the build
    # interpreter: the fairseq setup then omits its optional PyTorch/CUDA training ops.
    wheelhouse="$prefix/dpg-wheels"
    mkdir -p "$wheelhouse"
    if ! compgen -G "$wheelhouse/fairseq-0.12.2-*.whl" >/dev/null; then
        build="$prefix/dpg-build"
        "$py" -m venv "$build"
        "$build/bin/python" -m pip install 'pip==24.0' 'setuptools==68.2.2' 'wheel==0.41.3' 'Cython==0.29.37' 'numpy==1.26.4'
        "$build/bin/python" -m pip wheel --no-cache-dir --no-build-isolation --no-deps --wheel-dir "$wheelhouse" 'fairseq==0.12.2'
    fi
    "$py" -m pip install -c "$pins" -r requirements-dpg.txt "$wheelhouse"/fairseq-0.12.2-*.whl
    "$py" -m pip check
    "$py" -c 'from importlib.metadata import version; from modelscope.pipelines.multi_modal.visual_question_answering_pipeline import VisualQuestionAnsweringPipeline; print("MPLUG_PIPELINE_IMPORT_OK", version("modelscope"))'
    printf 'DPG_ENV_READY %s\n' "$prefix"
}
if [[ "$1" == setup || "$1" == setup-smoke ]]; then
    install_environment
    [[ "$1" != setup ]] || exit 0
fi

export SCD_RESOURCES="${SCD_RESOURCES:-$afs/workspace/DPO-mvpo-multiseed-v1/configs/resources.json}"
export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD"
export GIT_CONFIG_KEY_1=safe.directory
GIT_CONFIG_VALUE_1=$(PYTHONPATH="$PWD/src" "$py" -c 'from self_correct_dpo.resources import load_resources; print(load_resources()["DPG_REPO"])')
export GIT_CONFIG_VALUE_1
if [[ "$1" == dpg-group ]]; then
    export SCD_EVAL_GPUS="${SCD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
    exec bash scripts/run_reuse_composition.sh "$@"
fi

export SCD_EVAL_GPUS="${SCD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
[[ "$SCD_EVAL_GPUS" != *,* ]] || { echo 'Use one allocated GPU for smoke' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$SCD_EVAL_GPUS"
"$py" -I -u - <<'PY'
import torch
print("DPG runtime", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
print("CUDA_KERNEL_OK", torch.ones(4, device="cuda").sum().item(), flush=True)
PY

exec bash scripts/run_reuse_composition.sh dpg-smoke
