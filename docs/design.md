# 设计

本文件写多视图功能现在是什么样。为什么这样选见 `docs/decisions/`：输入约定见 `0001-unposed-image-set-input.md`，不训练的基线见 `0002-training-free-multiview-baseline.md`。怎么用、验证到了哪一步见 `docs/usage.md`，其中"验证状态"一节的表以那里为准。

目前只有不需要训练的基线，用的是官方 2.1 的几何和贴图权重，没有新的模型层、权重或配置。它只接入了 Python 接口和 `demo_multiview.py`，`gradio_app.py`、`api_server.py`、`model_worker.py` 和 cat3d 的调用方还是上游的单图行为。上游留下的 2.0 多视图代码（`DinoImageEncoderMV`、`MVImageProcessorV2`、gradio 的前后左右四个输入框）按固定方位写，本功能不用它们。

## 输入约定

- 一组图片是同一个物体的 1～4 张图，只出一个模型。图片没有视图名和方位编号，顺序不带含义。
- 张数上限是 4，两个入口各自检查：几何的 `hy3dshape.multiview_pipeline.MAX_IMAGES`，贴图的 `textureGenPipeline.MAX_REFERENCE_IMAGES`。超过时抛 `ValueError`，不截断，也不把一组拆成几个物体。4 是首版的实验预算，不代表官方权重最多支持 4 张，也不代表第 5 张就会失效，按服务器上的实测再调。
- 对输入图的要求：都是同一个物体；物体完整地在画面里；前景能分割出来（有透明通道，或者背景能被 rembg 抠掉）；尽量从互补的角度拍。几何的预处理照上游单图：每张图各自按透明通道找前景、居中、铺白底、缩放到 512，前景占画面的比例由官方配置的留边比例决定（`border_ratio`，上游默认 0.15）。
- 重复的图片保留，同一张图给两次相当于加大它的权重。
- 和"一批图"的区别：上游 `Hunyuan3DDiTFlowMatchingPipeline` 把图片列表当作一批互不相干的图，每张出一个模型。多视图管线绝不把整组图交给上游的 `prepare_image()`，而是逐张预处理，整组只建一个 latent。
- 输出坐标系照上游 2.1，不另加旋转，也不按某张输入图的方向转模型。

## 几何：`Hunyuan3DUnposedMultiViewPipeline`

在 `hy3dshape/hy3dshape/multiview_pipeline.py`，继承 `Hunyuan3DDiTFlowMatchingPipeline`，加载（`from_pretrained`、`from_single_file`）原样继承，权重目录、`config.yaml` 字段和 state dict 键名都不变。

```python
pipeline(images, *, fusion="velocity_mean", num_inference_steps=50, guidance_scale=5.0, generator=None,
         octree_resolution=384, num_chunks=8000, output_type="trimesh", enable_pbar=True)
```

**单图直接回父类。** 检查完参数后，只有一张图时（不论 `fusion` 是什么）直接调用 `Hunyuan3DDiTFlowMatchingPipeline.__call__(image=..., ...)`，在此之前不预处理、不建 latent、不消耗随机数。路径仍以路径传入（`Path` 转成 `str`），PIL 仍以 PIL 传入，所以结果和上游单图一样。

**多图的公共步骤：**

1. 逐张调用 `prepare_image(one_image)`，得到 `(1, 3, 512, 512)` 的图和掩码。
2. 按预处理后像素的字节内容给这几张图排序，固定处理顺序。排序只是为了让同一组图不管怎么排列，浮点运算的顺序都一样，结果可复现；顺序不传给模型。
3. 逐张调用 `encode_cond()` 编码一次，每张得到 `c_i`，形状 `(1, L, C)`。官方配置下 DINOv2 输入 518，`L = 37² + 1 = 1370`（含 CLS），`C = 1024`。不加视图编号的位置编码。无条件 token 用 `conditioner.unconditional_embedding(1)`，即单图的全零 token。
4. 采样步骤照父类：`sigmas = np.linspace(0, 1, num_inference_steps)`，交给模型的时间是 `t / num_train_timesteps`，整组只建一个形状为 `(1, *vae.latent_shape)` 的 latent，每个时间步只调用一次 `scheduler.step`。
5. 导出照父类的默认值：`box_v=1.01`、`mc_level=0.0`、`mc_algo=None`。

**两种融合。** 记 `v_i = DiT(x_t, t, c_i)`，`u` 是同一 `x_t, t` 下的无条件预测，`s` 是 `guidance_scale`。CFG 的开关照父类：`s >= 0` 且模型没有 `guidance_embed` 时开。

| | `velocity_mean`（默认） | `token_concat` |
|---|---|---|
| 条件 | 每次前向只送一张图的 `c_i`，都用同一个 `x_t` | `C = cat(c_1, ..., c_N)`，沿序列维拼接，`(1, N·L, 1024)` |
| CFG 开 | `v = u + s · (mean_i(v_i) − u)`，等于各图引导后速度的平均 | 无条件分支把单图的全零 token 重复 N 次，`(1, N·L, 1024)`，和 `C` 沿 batch 维拼成 `(2, N·L, 1024)`；latent 也复制成两份；`v = v_uncond + s · (v_cond − v_uncond)` |
| CFG 关 | `v = mean_i(v_i)` | 只用 `C` 和一份 `x_t` |
| 每步 DiT 前向 | CFG 开时 N+1 次，CFG 关时 N 次，都是 batch 1；`u` 由各图共用 | 1 次；CFG 开时 batch 2，关时 batch 1 |

`velocity_mean` 用 float32 边算边累加，不同时保留 N 份预测，平均和引导做完后转回 latent 的 dtype 再交给 scheduler。`token_concat` 能用，是因为 `HunYuanDiTPlain` 的交叉注意力按实际的条件长度构造 K/V；官方配置关闭了固定长度的 `AttentionPool`（`use_attention_pooling: false`）和 decoupled CA。模型开着 `use_attention_pooling` 时，`token_concat` 抛 `RuntimeError`；条件编码器返回的不只是 `main` 一路（比如 `DualImageEncoder`）时，多图路径也抛 `RuntimeError`，提示改用官方 `hunyuan3d-dit-v2-1`。管线如果是用上游的固定视图组件（`MVImageProcessorV2` 或 `DinoImageEncoderMV`）加载的，不管给几张图，一开始就抛 `RuntimeError`，同样提示改用官方 `hunyuan3d-dit-v2-1`。

读不出来或者预处理失败的图（路径不存在、文件损坏、前景为空）抛 `ValueError`，报出是第几张图和它的路径。只给一张图时，这个判断只在上游管线已经报错之后才做，正常的单图不会被多读一遍，模型和运行环境的错误原样抛出。

**成本。** 按 DiT 处理的样本数算，`velocity_mean` 开 CFG 时约是单图（一次 batch 2 前向）的 `(N+1)/2` 倍，4 张约 2.5 倍；关 CFG 时是 N 倍。同一时刻只有一份 batch 1 前向的显存工作区、一份累加器和 N 份条件（FP16 每份约 2.7 MiB）。`token_concat` 每步只跑一次 DiT，但交叉注意力的 K/V 长度随 N 线性增长，4 张是 5480 个条件 token。实际耗时和显存要在服务器上量。

## 贴图：多张参考图

改了两个上游文件，其余照旧：

- `hy3dpaint/textureGenPipeline.py`：`Hunyuan3DPaintPipeline.__call__` 的 `image_path` 除了一个 `str` 路径或一张 PIL 图，现在也接受 1～4 个路径（`str` 或 `Path`）或 PIL 图组成的 list/tuple，以及单个 `Path`（上游传列表会因 `image_prompt` 未赋值而报错）。路径在函数里读进来并关闭文件，保留原来的 mode。末尾新增 `seed=0`，传给 `multiviewDiffusionNet`。之后每张参考图照上游的做法缩放到 512、按透明通道铺白底、转 RGB。
- `hy3dpaint/utils/multiview_utils.py`：`multiviewDiffusionNet.__call__` 和 `forward_one` 末尾新增 `seed=0`，原来写死的 `seed_everything(0)` 和扩散的 `Generator(...).manual_seed(0)` 改用它。多张参考图时先按缩放后图片的像素字节排序，然后全部参考图都进参考分支（`self.pipeline(input_images, ...)`）和 DINOv2（`self.dino_v2(input_images)`）。一张时 DINOv2 仍按上游的写法 `self.dino_v2(input_images[0])` 调用。

张量的几个轴要分清：`B = 1` 是一个物体；`N_ref` 是参考图的张数（1～4）；`N_gen` 是为烘焙选出的渲染视角数（`max_num_view`，默认最多 6 个），它和参考图张数无关。

- 参考分支：每张参考图经 VAE 编码，拼成 `(1, N_ref, C, h, w)` 的 `ref_latents`；参考 UNet 按 `N_ref` 展开，各层的参考特征按 `(b n) l c -> b (n l) c` 写进缓存，生成侧的参考注意力因此同时看到全部参考图。CFG 复制的是 batch 轴，`N_ref` 保持完整。
- DINOv2（`facebook/dinov2-giant`）：多张时输出 `(N_ref, L, D)`，重排成 `(1, N_ref·L, D)`，经 `ImageProjModel` 每个 token 投成 4 个 context token，再按 PBR 材质和生成视角复制。这里的 `L` 是贴图 DINOv2 的 token 数，不是几何的 1370。
- `camera_azims`、`num_in_batch` 都是生成视角的参数，和参考图编号无关。

上游发布的贴图训练代码每次只用一张参考图，没有联合多张参考图的训练。官方贴图权重是否做过这样的训练没有公开，多张参考图的效果没有验证过，要实测。

## 随机数

- 几何：调用方传 `generator`（`demo_multiview.py` 用 `torch.manual_seed(seed)`），只用于初始 latent。多图路径不再消耗别的随机数。
- 贴图：`seed` 同时设置 Python、NumPy、PyTorch 的全局随机数（`seed_everything`）和扩散采样的 `Generator`。参考图经 VAE 编码时 `posterior.sample()` 用的是全局随机数，每张参考图各取一次，这就是多张参考图要先固定处理顺序的原因。`seed` 默认 0，和上游写死的值一样。
- `demo_multiview.py` 的 `--seed` 同时用于几何和贴图，范围 0～2³²−1（NumPy 的限制）。

## 两个阶段的资源

`demo_multiview.py` 先用一个 rembg 会话给没有透明背景的图抠图，用完释放；再加载几何模型生成 `shape.glb`，然后删掉几何管线、`gc.collect()`、`torch.cuda.empty_cache()`，之后才加载贴图模型。几何和贴图的模型不同时留在显存里。

## 权重兼容性

本功能不改模型结构、权重格式、配置字段和训练数据格式，直接加载官方 `tencent/Hunyuan3D-2.1` 的 `hunyuan3d-dit-v2-1` 和 `hunyuan3d-paintpbr-v2-1`。没有本项目自己训练的权重，也还没有多视图模型的子目录名。

## 验证状态

见 `docs/usage.md` 的"验证状态"一节。目前全部未在真实权重上运行过。
