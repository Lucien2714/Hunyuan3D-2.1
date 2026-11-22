# 使用说明

本文件说明多视图功能怎么装、怎么跑，以及各项能力在真实权重上的验证状态。多视图输入的约定和内部做法见 `docs/design.md`。

目前只有不需要训练的多图基线：几何用官方 2.1 单图权重做多图融合，贴图让官方贴图权重同时看到全部参考图。它只接入了 Python 接口和命令行 `demo_multiview.py`，`gradio_app.py`、`api_server.py` 还是上游的单图行为。本项目还没有训练命令，也没有自己训练的权重。

## 环境

和上游一样，只能在有 CUDA 的 Linux 机器上完整运行。上游用 Python 3.10、PyTorch 2.5.1+cu124 测试，依赖版本锁在 `requirements.txt`。在仓库根目录执行：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cd hy3dpaint/custom_rasterizer && pip install -e . && cd ../..
cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh && cd ../..
```

`--shape-only` 只跑几何，不需要后两个扩展；完整流程两个都要。多视图功能没有新增依赖。

## 权重和缓存

多视图基线不改权重格式和官方的查找逻辑，用到的都是上游已有的权重。它们分在几个地方，只设置 `$HY3DGEN_MODELS` 不能离线跑完完整流程：

| 用途 | 谁加载 | 查找位置 | 说明 |
|---|---|---|---|
| 几何 DiT、VAE、条件编码器 | `Hunyuan3DUnposedMultiViewPipeline.from_pretrained("tencent/Hunyuan3D-2.1")`，继承自上游 | `$HY3DGEN_MODELS/tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/`，`$HY3DGEN_MODELS` 默认 `~/.cache/hy3dgen` | 目录里是 `config.yaml` 和 `model.fp16.ckpt`。目录不存在时上游代码会用 `huggingface_hub.snapshot_download` 下载到这里。按项目约定优先从 ModelScope 下载后放到这个目录；ModelScope 上的仓库名本 PR 没有在服务器上核实过。 |
| 几何的 DINOv2 结构 | 上游 `ImageEncoder`，`AutoModel.from_pretrained(version)` | Hugging Face 缓存（`$HF_HOME`，默认 `~/.cache/huggingface`） | 仓库里的几何训练配置写的是 `version: facebook/dinov2-large`。如果官方 `config.yaml` 也这样写，加载时会先取这个模型，再用 `model.fp16.ckpt` 里的条件编码器权重覆盖。 |
| 贴图扩散模型 | 上游 `multiviewDiffusionNet`，`huggingface_hub.snapshot_download("tencent/Hunyuan3D-2.1", allow_patterns=["hunyuan3d-paintpbr-v2-1/*"])` | Hugging Face 缓存 | 不走 `$HY3DGEN_MODELS`。国内服务器需要能访问 Hugging Face 或镜像（`HF_ENDPOINT`），或者事先把它放进 Hugging Face 缓存。 |
| 贴图的 DINOv2 | 上游 `Dino_v2("facebook/dinov2-giant")` | Hugging Face 缓存 | 同上。 |
| 超分 | 上游 `imageSuperNet` | `hy3dpaint/ckpt/RealESRGAN_x4plus.pth` | 要手动下载：`wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt`。`demo_multiview.py` 不带 `--shape-only` 时，开始前会检查这个文件在不在。 |
| 抠图 | 上游 `BackgroundRemover`（rembg 默认的 u2net） | `~/.u2net/` | 只有输入图没有可用的透明通道时才用，第一次用时 rembg 会从 GitHub 下载。 |

## 命令行

在仓库根目录运行：

```bash
python demo_multiview.py --images a.png b.png c.png --output-dir outputs/run1
```

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--images PATH [PATH ...]` | 必填 | 同一物体的 1～4 张图，顺序和拍摄方向都不用管。超过 4 张时在读图和加载模型之前就报错。 |
| `--fusion {velocity_mean,token_concat}` | `velocity_mean` | 几何的多图融合方式，见 `docs/design.md`。只有一张图时不起作用。 |
| `--seed INT` | `0` | 几何和贴图共用的种子，范围 0～2³²−1。 |
| `--output-dir DIR` | 必填 | 本次运行的输出目录。目录可以已经存在，但里面不能有上一次运行留下的同名产物，否则开始前就报错并列出这些文件。 |
| `--shape-only` | 关 | 只跑几何，输出 `shape.glb` 后结束，不加载贴图模型。 |

运行顺序和输出（都在 `--output-dir` 下）：

1. 逐张读图，按透明信息分三种情况。透明信息可以是透明通道，也可以是 PNG 的调色板透明或透明色键（RGB、灰度图用一种颜色表示透明）：
   - 全透明的图直接报错并给出文件名，不会丢掉透明通道再去抠图。
   - 已经有透明背景的图（有不是完全不透明的像素）保留它的前景，不再抠图。
   - 没有透明信息或者完全不透明的图，用同一个 rembg 会话逐张抠图。

   处理完没有前景的图报错并给出文件名。

   PNG 的透明色键（灰度 1、2、4、8、16 位，RGB 8、16 位）按文件原始的位深匹配：一个像素的各个样本和色键完全相等才算透明，匹配完再把 16 位的颜色取高 8 位转成 8 位。所以 16 位的图里和色键只差低 8 位的像素仍然是不透明的前景。上面三种情况的判断和保留下来的前景用的是这同一份遮罩。无损压缩工具会把普通的带透明通道的图存成带色键的文件，用 ZopfliPNG 的默认优化试过，上面七种都能得到。

   脚本自己匹配色键，是因为 Pillow 转出的透明通道对其中几种文件不对（在 Pillow 10.4.0、11.3.0 和 12.3.0 上查证）：2 位和 4 位灰度，三个版本都把像素展开成 8 位、色键却不换算；1 位灰度，10.4.0 和 11.3.0 不换算色键，12.3.0 换算；16 位灰度读进来样本是完整的，但转换时大于 255 的值都被截成 255；16 位 RGB 读进来时每个样本只留下高 8 位，色键仍是 16 位的值。8 位灰度和 8 位 RGB 的色键 Pillow 处理得对，脚本的结果和它一样。16 位 RGB 的完整样本用 OpenCV 读，它是 `requirements.txt` 里已有的依赖。
2. 加载几何模型，用 `torch.manual_seed(seed)` 生成一个网格，写 `shape.glb`。网格为空时报错。之后释放几何模型和显存，再加载贴图模型。
3. 贴图（`max_num_view=6`、`resolution=512`，和 `demo.py` 一样），全部输入图都作为参考图。上游的贴图管线会先把 `shape.glb` 重新网格化成 `white_mesh_remesh.obj`，然后写 `textured.obj`、`textured.mtl`、`textured.jpg`（albedo）、`textured_metallic.jpg`、`textured_roughness.jpg`。
4. 用上游的 `create_glb_with_pbr_materials` 打包成 `model.glb`（base color 加 metallic-roughness 贴图），打包后检查每个图元都带这两张贴图，否则报错。打包用的临时文件在输出目录下的临时子目录里，结束后自动删除。

几何和贴图两个阶段都从加载模型之前开始计时，并在那时重置显存峰值。阶段结束时打印整个阶段（加载模型加推理）的耗时和 `torch.cuda.max_memory_allocated()`，中间另打印一行模型加载完成时已经用掉的时间。阶段中途失败（比如加载时显存不足）时，先打印失败的阶段名、已用时间和峰值，再照常报错。这里的峰值是 PyTorch 分配器的峰值，不等于整张卡的显存占用。上游说单图几何要 10 GB、贴图要 21 GB，多图的显存没有实测过。

## Python 接口

在仓库根目录运行：

```python
import gc
import sys
sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")

import torch
from hy3dshape.multiview_pipeline import Hunyuan3DUnposedMultiViewPipeline

shape = Hunyuan3DUnposedMultiViewPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
mesh = shape(["a.png", "b.png"], fusion="velocity_mean", generator=torch.manual_seed(0))[0]
mesh.export("shape.glb")

# Free the geometry models before the texture models are loaded.
del shape, mesh
gc.collect()
torch.cuda.empty_cache()

# Must run before the texture pipeline imports its super-resolution dependencies (basicsr), as in demo.py.
from torchvision_fix import apply_fix
apply_fix()
from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
config.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"  # as in demo.py
paint = Hunyuan3DPaintPipeline(config)
paint(mesh_path="shape.glb", image_path=["a.png", "b.png"], output_mesh_path="textured.obj", save_glb=False, seed=0)
```

- `Hunyuan3DUnposedMultiViewPipeline.__call__(images, *, fusion="velocity_mean", num_inference_steps=50, guidance_scale=5.0, generator=None, octree_resolution=384, num_chunks=8000, output_type="trimesh", enable_pbar=True)`。`images` 是一个路径（`str` 或 `Path`）或 PIL 图，或者 1～4 个它们组成的 list/tuple，表示同一个物体。返回只有一个元素的网格列表；`output_type="latent"` 返回形状为 `(1, L, C)` 的 latent。只给一张图时直接调用上游 `Hunyuan3DDiTFlowMatchingPipeline.__call__`。
- 抠图不在几何管线里。没有透明背景的图要先抠图，和上游单图一样；`demo_multiview.py` 会自动做这一步。
- `Hunyuan3DPaintPipeline.__call__` 的 `image_path` 现在也接受 1～4 个路径或 PIL 图组成的 list/tuple，末尾新增可选参数 `seed=0`。只传一个路径或一张 PIL 图、不传 `seed` 时和上游一样。
- 两个管线各自限制最多 4 张，超过时抛 `ValueError`。

## 服务器验证命令

以下命令在装好依赖、扩展和权重的 Linux CUDA 服务器上，在包含本功能的提交的仓库根目录执行，所有代码块按顺序在同一个 bash 里运行。结果写进 `mktemp -d` 建的新目录。建议同时记下 GPU 型号、驱动、Torch/CUDA 版本、两个提交、权重版本和全部参数。

### 1. 多图端到端

```bash
export ROOT="$(pwd -P)"
export RUN="$(mktemp -d)"
export D="$ROOT/hy3dshape/tools/mini_trainset/preprocessed/00a4cff37043361068376104a292f5b44b5eacbd174651553b6a7ae35647a2a6/render_cond"
export CUDA_VISIBLE_DEVICES=0
git rev-parse HEAD
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
printf 'Results: %s\n' "$RUN"

views=("$D/000.png" "$D/006.png" "$D/012.png" "$D/018.png")

python demo_multiview.py --images "${views[0]}" --seed 1234 --output-dir "$RUN/one"

for fusion in velocity_mean token_concat; do
  python demo_multiview.py --images "${views[@]}" --seed 1234 \
    --fusion "$fusion" --output-dir "$RUN/$fusion"
done

python - <<'PY'
import os
from pathlib import Path
from pygltflib import GLTF2

for name in ("one", "velocity_mean", "token_concat"):
    p = Path(os.environ["RUN"]) / name / "model.glb"
    g = GLTF2().load(str(p))
    for mesh in g.meshes:
        for prim in mesh.primitives:
            assert prim.material is not None, p
            m = g.materials[prim.material].pbrMetallicRoughness
            assert m.baseColorTexture is not None and m.metallicRoughnessTexture is not None, p
    assert g.meshes and g.images, p
    print(name, p.stat().st_size, "bytes", len(g.images), "images")
PY
```

`000/006/012/018` 只是可复现的取样编号，不代表四个固定方位。4 张如遇显存不足，记下失败的阶段、卡型、打印出的峰值和张数，再用 `000.png 006.png` 两张复跑。不要换融合方式、减少参考图或退回白模后仍记成成功。

### 2. 单图结果和改动前是否一致

几何和贴图分开比。几何用同一张图、同一个显式 generator，比较最终 latent 和导出的网格。贴图用样例里的同一个输入网格和同一张 PIL 图，排除几何差异的影响；改动前的贴图内部写死种子 `0`，所以这里也用 `0`。改动后的贴图分别用旧的调用方式（传一张图）和新的列表方式跑。改动前的源码用 `git archive` 从 `c5774b8` 取出，两版共用同一个运行环境、权重和已经编译好的扩展。

```bash
export BASE="$RUN/base-src"
mkdir "$BASE"
git archive c5774b8 hy3dshape/hy3dshape hy3dpaint/textureGenPipeline.py \
  hy3dpaint/hunyuanpaintpbr hy3dpaint/utils hy3dpaint/DifferentiableRenderer \
  torchvision_fix.py | tar -x -C "$BASE"
for so in "$ROOT"/hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor*.so; do
  test -f "$so"
  ln -s "$so" "$BASE/hy3dpaint/DifferentiableRenderer/$(basename "$so")"
done

cat > "$RUN/single.py" <<'PY'
import os
import sys
from pathlib import Path

src, task, mode, label = sys.argv[1:]
src = Path(src)
sys.path[:0] = [str(src / "hy3dshape"), str(src / "hy3dpaint"), str(src)]
import numpy as np
import torch
from PIL import Image

d = Path(os.environ["D"])
out = Path(os.environ["RUN"]) / label
out.mkdir()
with Image.open(d / "000.png") as f:
    image = f.convert("RGBA")

if task == "shape":
    if mode == "upstream":
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline as Pipe
        args = {"image": image}
    else:
        from hy3dshape.multiview_pipeline import Hunyuan3DUnposedMultiViewPipeline as Pipe
        args = {"images": [image], "fusion": mode}
    pipe = Pipe.from_pretrained("tencent/Hunyuan3D-2.1")
    z = pipe(**args, generator=torch.manual_seed(0), output_type="latent")
    np.save(out / "latent.npy", z.detach().cpu().numpy())
    mesh = pipe(**args, generator=torch.manual_seed(0), output_type="trimesh")[0]
    np.savez(out / "mesh.npz", v=mesh.vertices, f=mesh.faces)
else:
    from torchvision_fix import apply_fix
    apply_fix()
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    cfg = Hunyuan3DPaintConfig(6, 512)
    cfg.realesrgan_ckpt_path = str(Path(os.environ["ROOT"]) / "hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
    pipe = Hunyuan3DPaintPipeline(cfg)
    args = {"image_path": image} if mode == "upstream" else {"image_path": [image], "seed": 0}
    pipe(mesh_path=str(d / "mesh.ply"), output_mesh_path=str(out / "textured.obj"),
         use_remesh=False, save_glb=False, **args)
PY

python "$RUN/single.py" "$BASE" shape upstream old-shape
python "$RUN/single.py" "$ROOT" shape upstream new-upstream
python "$RUN/single.py" "$ROOT" shape velocity_mean new-mean
python "$RUN/single.py" "$ROOT" shape token_concat new-concat
python "$RUN/single.py" "$BASE" paint upstream old-paint
python "$RUN/single.py" "$ROOT" paint upstream new-scalar-paint
python "$RUN/single.py" "$ROOT" paint list new-list-paint

python - <<'PY'
from pathlib import Path
import os
import numpy as np
from PIL import Image

r = Path(os.environ["RUN"])
def identical(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()

z = np.load(r / "old-shape/latent.npy")
with np.load(r / "old-shape/mesh.npz") as a:
    for name in ("new-upstream", "new-mean", "new-concat"):
        assert identical(z, np.load(r / name / "latent.npy")), (name, "latent")
        with np.load(r / name / "mesh.npz") as b:
            for key in ("v", "f"):
                assert identical(a[key], b[key]), (name, key, a[key].shape, b[key].shape)
        print(name, "identical latent, vertices and faces")

for name in ("new-scalar-paint", "new-list-paint"):
    for suffix in (".jpg", "_metallic.jpg", "_roughness.jpg"):
        file = "textured" + suffix
        with Image.open(r / "old-paint" / file) as a, Image.open(r / name / file) as b:
            assert identical(np.asarray(a), np.asarray(b)), (name, file)
    assert (r / "old-paint/textured.obj").read_bytes() == (r / name / "textured.obj").read_bytes(), name
    print(name, "identical OBJ and decoded albedo, metallic, roughness")
PY
```

**怎样算没变。** 上面的逐位比较全部通过，才记"单图逐位一致"。GLB 文件的哈希不作为判据。如果没通过，先用新的输出标签在旧源码上把同一条命令再跑一次，看旧对旧是否稳定。latent 相同而顶点或贴图不同时，分别检查表面提取、UV 展开和烘焙，记下第一个不同的地方、顶点数和面数、坐标的最大绝对误差或像素误差。旧对旧本身就有波动时，只能按实测写可重复到什么程度，不能放宽标准后仍记"逐位一致"。`demo.py`、`model_worker.py` 都不把固定的 generator 传给几何，HTTP 的 `seed` 字段也代替不了这组命令。

### 3. 换序后结果是否一致

一组图片的顺序不带含义，管线按图片内容固定处理顺序，所以把同一组图倒过来传，几何结果应当逐位相同。

```bash
for fusion in velocity_mean token_concat; do
  python demo_multiview.py --images "${views[3]}" "${views[2]}" "${views[1]}" "${views[0]}" --seed 1234 \
    --fusion "$fusion" --shape-only --output-dir "$RUN/$fusion-reversed"
done

python - <<'PY'
import os
from pathlib import Path
import trimesh

r = Path(os.environ["RUN"])
for fusion in ("velocity_mean", "token_concat"):
    a = trimesh.load(r / fusion / "shape.glb", force="mesh", process=False)
    b = trimesh.load(r / f"{fusion}-reversed" / "shape.glb", force="mesh", process=False)
    same = a.vertices.tobytes() == b.vertices.tobytes() and a.faces.tobytes() == b.faces.tobytes()
    print(fusion, "reversed order identical" if same else "reversed order DIFFERENT")
PY
```

### 4. 多图比单图好多少

先对第 1 组的三个结果和样例的真值网格算倒角距离：

```bash
PYTHONPATH="$ROOT/hy3dshape/tools/evaluation" python - <<'PY'
import os
from pathlib import Path
import numpy as np
import trimesh
from chamfer_distance import chamfer_distance_from_meshes

d, r = Path(os.environ["D"]), Path(os.environ["RUN"])
gt = trimesh.load(d / "mesh.ply", force="mesh", process=False)
scores = {}
for run in ("one", "velocity_mean", "token_concat"):
    mesh = trimesh.load(r / run / "shape.glb", force="mesh", process=False)
    np.random.seed(2026)
    metrics = chamfer_distance_from_meshes(mesh.vertices, mesh.faces, gt.vertices, gt.faces,
                                          n_samples=100000)
    scores[run] = metrics["cd_l2"]
    print(run, metrics, "vertices", len(mesh.vertices), "faces", len(mesh.faces))
for run in ("velocity_mean", "token_concat"):
    print(run, "single CD minus multi CD", scores["one"] - scores[run])
PY
```

`cd_l2` 越低越好，打印的差值为正表示多图更好。这个函数返回的是双向、未平方的平均最近邻距离。它对每个网格**逐轴独立**归一化到 `[-1,1]`，会掩盖尺度和长宽高比例的错误，也不会自动对齐朝向，所以还要另看原始包围盒比例、侧面和背面的轮廓以及朝向。`is_watertight.py` 写死了 `your_mesh.glb`，直接运行评估不到本次产物，而且水密性也代替不了形状相似度。

正式汇报时，把上面的对照扩到 `mini_trainset` 的全部 8 个物体、同一组种子，几何批量实验可以加 `--shape-only`。至少把四张入选图各自单独跑一遍单图，报告多图相对单图均值的配对差，并列出相对最好的单图的差、变差的物体数、耗时和显存，这样才能分清是多图真有帮助，还是恰好 `000.png` 比较差。贴图另外固定同一个 `mesh.ply`、种子和渲染视角，比较单参考图、四张参考图、倒序的四张参考图，重点看只在后几张图里出现的标记、背面配色和材质，再看没有作为输入的那些渲染图。比较两个几何不同的最终 GLB 只能说明端到端的差别，不能单独归因于贴图。8 个样例只够判断值不值得继续投入，不足以说明普遍有提升。

## 验证状态

本项目没有测试，代码在真实权重上跑过才算验证过。下表逐项记录，没有在真实权重上跑过的标"未运行"。

| 能力 | 输入与配置 | 状态 | 机器及环境 | 提交与权重 | 结果或失败点 |
|---|---|---|---|---|---|
| 单图几何回归（新管线与上游逐位比较） | `000.png`，种子 0，`velocity_mean` 和 `token_concat` 各一次，第 2 组命令 | 未运行 | — | — | — |
| 多图几何 `velocity_mean` | 4 张，种子 1234，第 1 组命令 | 未运行 | — | — | — |
| 多图几何 `token_concat` | 4 张，种子 1234，第 1 组命令 | 未运行 | — | — | — |
| 换序后几何一致 | 4 张倒序，种子 1234，第 3 组命令 | 未运行 | — | — | — |
| 单图贴图回归（旧调用方式和列表方式） | `mesh.ply` + `000.png`，种子 0，第 2 组命令 | 未运行 | — | — | — |
| 多参考图贴图 | 4 张参考图，第 1 组命令 | 未运行 | — | — | — |
| 端到端 PBR GLB（`demo_multiview.py`） | 1 张和 4 张，第 1 组命令 | 未运行 | — | — | — |
| `mini_trainset` 配对质量 | 倒角距离，第 4 组命令 | 未运行 | — | — | — |
| 峰值显存 | 1、2、4 张，几何和贴图分别记录 | 未运行 | — | — | — |
