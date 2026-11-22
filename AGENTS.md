# hunyuan3d-2.1-mv

给 Hunyuan3D 2.1 加上多视图输入：输入同一物体的几张图片，视角不固定，也不用标注每张图是从哪个方向拍的，输出带贴图的 3D 模型，目的是让结果和每一张输入图都对得上。几何模型（Hunyuan3D-Shape-v2-1）和贴图模型（Hunyuan3D-Paint-v2-1）都要支持。要做的包括不用训练的多图融合基线、多视图的条件编码、训练或微调、几何的推理管线、贴图管线的多参考图输入，以及让 `gradio_app.py`、`api_server.py` 能用上多视图。生成的模型会被 cat3d 的 `hunyuan3d_server` 使用，cat3d 是另一个仓库。

Hunyuan3D 2.0 的多视图版本 Hunyuan3D-2mv 用的是固定的正、左、后、右四个视图。本项目不沿用这一点，原因见 `docs/decisions/0001-unposed-image-set-input.md`。

本仓库 fork 自 [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)。`README.md` 是上游的英文说明，不要改它，免得和上游合并时冲突。本项目自己的文档放在 `docs/` 下，动手前按要改的部分读对应文档：

- `docs/usage.md`：多视图模型怎么装、怎么推理、怎么训练，配置项，验证状态。
- `docs/design.md`：多视图条件怎么进模型、多视图输入的约定、权重和配置的格式、和上游单视图的差别。
- `docs/deploy-notes.md`：那台国内 GPU 服务器上的目录、下载源和踩过的坑。服务器本身的情况先看 cat3d 仓库里的 `docs/deploy-notes.md`，这里只记本项目特有的。
- `docs/decisions/`：决策记录，说明某个做法当时为什么这样选。

**`docs/usage.md`、`docs/design.md` 和 `docs/deploy-notes.md` 还没有建立。** 第一个需要它们的 PR 负责创建，之后按"文档"一节维护。

## 上游已有的多视图基础

上游的几何代码里留着 2.0 多视图版本的一部分，2.1 的官方权重没有用到它们。它们是按固定四视图写的，和本项目的约定不同（见"多视图约定"），只能参考，不能当作本项目的约定。动手前先读：

- `hy3dshape/hy3dshape/models/conditioner.py` 的 `DinoImageEncoderMV`：每张视图分别过 DINOv2，加一个按视图编号的正弦位置编码（`view_embed`），再把各视图的 token 拼成一串。`unconditional_embedding()` 按视图数给出全零条件。
- `hy3dshape/hy3dshape/preprocessors.py` 的 `MVImageProcessorV2`：输入 `{视图名: 图片}` 的字典，按 `front=0, left=1, back=2, right=3` 排序，返回 `view_idxs`。注释写的是"正面、正面顺时针转 90°、背面、正面顺时针转 270°"。
- `hy3dshape/hy3dshape/pipelines.py` 里登记了 `Hunyuan3D-2mv` 对应的 2.0 VAE。
- `gradio_app.py` 有前、后、左、右四个多视图输入框，拼成同样的字典。

这些代码在 2.1 上从来没有跑通过，不能当作"已经能用"。

另外，`hy3dshape/hy3dshape/pipelines.py` 的 `prepare_image()` 把传入的图片列表当作一批互不相干的图，每张图各出一个模型。本项目说的"一组图片"是同一个物体、只出一个模型，所以多视图输入不能直接复用这个列表入口。

上游的贴图代码在结构上能接多张参考图，但实际只用了第一张：

- `hy3dpaint/hunyuanpaintpbr/unet/modules.py` 的参考分支按参考图的张数（`N_ref`）处理，各张参考图的特征拼在一起交给参考注意力。
- `hy3dpaint/utils/multiview_utils.py` 的 `multiviewDiffusionNet.forward_one()` 只把第一张图传给扩散管线和 DINOv2。
- `hy3dpaint/textureGenPipeline.py` 的 `Hunyuan3DPaintPipeline.__call__()` 在传入图片列表时会报错，因为 `image_prompt` 没有赋值。
- 上游发布的贴图训练代码每次只用一张参考图（`hy3dpaint/hunyuanpaintpbr/unet/model.py` 的 `prepare_batch_data()`）。官方贴图权重实际是怎么训练的没有公开；按这份代码，多张参考图是没有训练过的用法。

## 命令

上游官方的安装和运行方式，只能在有 CUDA 的 Linux 机器上完整运行。上游用 Python 3.10、PyTorch 2.5.1+cu124 测试：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cd hy3dpaint/custom_rasterizer && pip install -e . && cd ../..
cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh && cd ../..

python demo.py                                   # 单图生成几何和贴图
python gradio_app.py --model_path tencent/Hunyuan3D-2.1 --subfolder hunyuan3d-dit-v2-1 \
  --texgen_model_path tencent/Hunyuan3D-2.1 --low_vram_mode
python api_server.py --port 8081                 # HTTP 接口，见 API_DOCUMENTATION.md
cd hy3dshape && bash train_demo.sh               # 几何模型训练，配置在 hy3dshape/configs/
```

本项目不写测试，也不设 CI。改动对不对，靠评审，以及在 GPU 服务器上用真实权重实际跑一遍来确认，见"验证状态要如实写"。开 PR 前只要求对改到的 Python 文件做一次语法检查。用 uv 里装好的 Python 3.10 跑，这样也能查出 3.10 不支持的写法，不用装依赖：

```powershell
uv run --no-project --python 3.10 python -m py_compile <改到的 .py 文件>
```

语法检查查不出用了更新版本的库才有的接口，这一点要自己按"代码规范"核对。

## 结构

- `hy3dshape/`：几何模型，本项目的主要改动在这里。
  - `hy3dshape/hy3dshape/models/conditioner.py`：图像条件编码（DINOv2）。多视图编码在这里。
  - `hy3dshape/hy3dshape/models/denoisers/hunyuandit.py`：2.1 的去噪网络 `HunYuanDiTPlain`，通过交叉注意力读条件 token。条件 token 变长（视图数 × 每张图的 token 数）要看这里能不能接住。
  - `hy3dshape/hy3dshape/models/diffusion/flow_matching_sit.py`：训练用的 Lightning 模块 `Diffuser`，负责编码条件、无条件分支和损失。
  - `hy3dshape/hy3dshape/models/autoencoders/`：形状 VAE（`ShapeVAE`），本项目不训练它。
  - `hy3dshape/hy3dshape/pipelines.py`：推理管线 `Hunyuan3DDiTFlowMatchingPipeline`，权重加载 `from_pretrained()`。
  - `hy3dshape/hy3dshape/preprocessors.py`：输入图的居中、缩放，以及上游按固定视图排序的 `MVImageProcessorV2`。抠图在 `hy3dshape/hy3dshape/rembg.py`。
  - `hy3dshape/hy3dshape/data/dit_asl.py`：训练数据加载。数据格式见 `hy3dshape/tools/README.md`，`hy3dshape/tools/mini_trainset/` 是随仓库带的几条样例数据。
  - `hy3dshape/main.py`、`hy3dshape/configs/`、`hy3dshape/scripts/train_deepspeed.sh`：训练入口、配置和多卡启动脚本。
  - `hy3dshape/tools/evaluation/`：倒角距离、水密检查。
- `hy3dpaint/`：PBR 贴图模型。它内部本来就先生成多个视角的图再烘焙成贴图，那是它的输出方式，和本项目的"多视图输入"不是一回事。本项目在这里要做的是让多张输入图都作为参考图。和参考图输入无关的部分，除非方案里写明，不改。
  - `hy3dpaint/textureGenPipeline.py`：贴图管线 `Hunyuan3DPaintPipeline`，负责选视角、调用多视角扩散、超分和烘焙。
  - `hy3dpaint/utils/multiview_utils.py`：`multiviewDiffusionNet`，把参考图、法线图和位置图交给扩散管线。
  - `hy3dpaint/hunyuanpaintpbr/`：扩散管线和 UNet。参考图经参考分支和 DINOv2 特征进入模型。
- 根目录的 `gradio_app.py`、`api_server.py`、`api_models.py`、`model_worker.py`、`demo.py`：演示和 HTTP 服务，都调用上面两个管线。

## 约定

**和上游的关系**

- `origin` 是本 fork，`upstream` 是腾讯的官方仓库，`upstream` 只用来拉取，推送地址已禁用。`gh` 的默认仓库已设为 `Lucien2714/Hunyuan3D-2.1`。在 fork 里执行 `gh pr create` 时，gh 可能默认把 PR 开到上游，所以 `gh` 命令一律带 `--repo Lucien2714/Hunyuan3D-2.1`，**绝不往上游开 PR、发评论或推送**。
- 同步上游用 `chore/sync-upstream` 分支：`git fetch upstream`，把 `upstream/main` 合并进来（merge，不 rebase），开 PR。
- 尽量不改上游文件。能加新文件、新类、新配置的，就不改原来的。不得不改上游文件时，改动越小越好，不顺手重排、重命名或格式化周围的代码。
- 上游的单视图行为保持不变：只给一张图时，`Hunyuan3DDiTFlowMatchingPipeline`、`Hunyuan3DPaintPipeline`、`gradio_app.py`、`api_server.py` 用官方 2.1 权重的结果要和改之前一样。
- 上游文件开头的许可声明照留，新文件不用加。许可的说法见 cat3d 仓库 `docs/backends.md` 的"许可"一节，商用前要自己确认。

**多视图约定**

- 多视图输入是同一物体的一组图片，1 到 N 张。图片没有视图名，也没有方位编号，顺序不带含义，调用方不需要知道每张图是从哪个角度拍的。N 的上限写在 `docs/design.md`。这样选的原因见 `docs/decisions/0001-unposed-image-set-input.md`。
- 不沿用 `MVImageProcessorV2` 的 `front=0, left=1, back=2, right=3`。模型里如果要区分图片，编号只表示"第几张"，不表示方位。
- 这套约定在训练数据、预处理、模型、几何和贴图的推理管线、`gradio_app.py`、`api_server.py` 和 cat3d 的调用方之间必须一致。要改的话，写决策记录，并同时改上面所有地方。
- 每张图各自抠图、居中、缩放，做法照上游单视图的预处理。对输入图的要求（背景、物体在画面里多大）写在 `docs/design.md`。
- 图片张数可以变。张数不同时条件 token 怎么排、无条件分支怎么给，在 `design.md` 里写清楚。只给一张图时，用官方 2.1 权重的结果要和上游单视图一样，见"和上游的关系"。
- 输出模型的坐标系照上游 2.1：多视图版本不另加旋转，也不按某一张输入图的拍摄方向去转模型。cat3d 那边对朝向的约定是 glTF 坐标系（Y 轴朝上，正面朝 +Z），如果多视图版本的朝向和单视图不同，要在文档里写明。

**权重、配置和数据**

- 本项目训练出的权重沿用上游的目录布局：`<模型目录>/<子目录>/config.yaml` 加 `model.fp16.ckpt`（或 `.safetensors`），这样 `from_pretrained()` 和 `--model_path`/`--subfolder` 不用改就能加载。多视图模型的子目录名定好后写进 `usage.md`。
- 权重优先从 ModelScope 下载，放到官方代码查找的本地位置（`$HY3DGEN_MODELS`，默认 `~/.cache/hy3dgen`），不改官方的查找逻辑。只在 Hugging Face 上有的权重要在文档里写明，国内服务器上 Hugging Face 的大文件基本下不来。
- 训练配置放在 `hy3dshape/configs/`，新文件名照现有的写法，写明编码器、图片张数上限和学习率。
- 权重、训练输出（`hy3dshape/output_folder/`）、数据集、`hy3dpaint/ckpt/`、`.env` 都不进 git。随仓库的样例数据只有上游那几条，不再加大文件。

**加或改一项多视图能力时要同步的地方**

1. 模型：`conditioner.py` 的编码器，`build_image_encoder()` 里的登记。
2. 训练：`dit_asl.py` 能读出同一物体的多张图，`Diffuser` 把它们传给编码器，`hy3dshape/configs/` 里有对应的配置。
3. 几何推理：推理管线接受一组图片，每张图各自预处理，无条件分支的长度和有条件的一致。
4. 贴图推理：`textureGenPipeline.py` 和 `multiview_utils.py` 把同一组图片都作为参考图。
5. 演示和服务：`gradio_app.py`、`api_server.py` 和 `api_models.py`（HTTP 接口的字段），`API_DOCUMENTATION.md`。
6. 文档：`usage.md`、`design.md`，以及 cat3d 那边要跟着改什么（写在 PR 描述里，cat3d 的改动在那个仓库里另开 PR）。

## 代码规范

下面的写法适用于本项目新写的文件。改上游文件时照那个文件已有的写法，只把下面"报错"和"不过度防御"两节用在自己写的那几行上。不要为了符合规范去改和当前任务无关的代码。

**兼容性**

- 代码要能在 Python 3.10 和上游锁定的依赖版本（`requirements.txt`：PyTorch 2.5.1、transformers 4.46.0、diffusers 0.30.0、pytorch-lightning 1.9.5）上运行。不用 3.11 以后才有的语法和标准库，不依赖更新版本的库才有的接口。
- 不为了方便加新依赖。确实要加时写明理由，写在 PR 描述里，并加进 `requirements.txt`。

**类型和数据**

- 新文件开头写 `from __future__ import annotations`，函数签名带类型标注。张量的形状写在 docstring 或注释里，例如 `(B, V, C, H, W)`。
- 配置照上游用 YAML + `instantiate_from_config()`，不另起一套配置系统。

**报错**

- 输入有问题（图片张数超过模型支持的、图片尺寸不对）时抛 `ValueError`，模型或运行环境出错时抛 `RuntimeError`。不要用 `assert` 检查用户输入，`python -O` 时它不生效。
- 报错信息要让用户知道下一步做什么：哪里出了问题，该检查哪个配置、哪个参数、哪份权重。
- 把一种异常转成另一种时用 `raise ... from exc`，保留原因。
- 不静默吞掉异常。不得不捕获 `Exception` 或忽略某个错误时，在旁边写注释说明原因。

**不过度防御**

- 只为实际会发生的情况写代码：用户会做的操作，模型、数据和文件系统真实会出的错。
- 只有人为构造才会出现的情况，不为它加检查、加锁或加回退分支。
- 例外是出错代价大的地方，这些照样要防：会覆盖已有的权重、训练输出或数据集，或者会让没有权限的人读到、改到东西（`api_server.py`）。
- 同一个条件只在一处检查。同一个进程里，调用方已经保证了的，被调用的函数不再检查一遍。从进程外进来的数据照样要检查：HTTP 请求、磁盘上的配置、权重和数据。

**注释和命名**

- 每个新模块开头有 docstring，说明它负责什么。
- 注释写原因，不复述代码做了什么。从上游或 2.0 官方代码照抄的部分，注释写明抄自哪个仓库的哪个文件。
- 只在模块内部用的函数和类，名字以下划线开头。
- 行宽不超过 120，中文字符算两列。

## 验证状态要如实写

本项目没有测试，代码在真实权重上跑过才算验证过，跑通了也不代表训练有效果。`docs/usage.md` 里要有一张表，逐项记录哪些在真实权重上跑过、在哪台机器、哪个提交、结果如何。

- 改了模型、训练或推理的代码后，如实更新那张表。没在真实权重上跑过的，标"未运行"。
- 改了模型、训练、推理、演示或 HTTP 服务的 PR，描述里要写出让用户在服务器上验证的命令：
  - 多视图怎么跑。
  - 单视图结果有没有变怎么比：同一张图、同一个种子，改前和改后的提交各跑一次。`demo.py`、`hy3dshape/minimal_demo.py` 和 `model_worker.py` 都不固定种子（`api_models.py` 有 `seed` 字段，但 `model_worker.py` 没用它），要直接调管线并传 `generator=torch.manual_seed(<种子>)`，或者在 gradio 界面里填种子。写明怎样算"没变"，比如顶点数和面数相同、倒角距离低于多少。
  - 改了训练的，用 `hy3dshape/tools/mini_trainset/` 跑几步确认能训练起来。注意 `hy3dshape/main.py` 的 `merge_cfg()` 会让配置文件里的 `training` 字段盖掉同名的命令行参数，所以 `--steps` 这类参数要改在配置里；`train_demo.sh` 写死了 8 张卡并带 `--deepspeed`；mini 过拟合配置的注释说它要 68G 显存。命令要对照服务器的卡数和显存写。
  - 只改演示和 HTTP 服务的，上游带的 `test_api_server.py` 可以作为验证命令的一部分。
- 训练结果要写明：用的数据（多少条、怎么渲染的）、步数、显卡、评估方法（`hy3dshape/tools/evaluation/` 的倒角距离等）和数字。只看过几张渲染图的，就写"只看过几张渲染图"。
- 开发机是 6 GB 显存的 Windows 机器。上游说几何生成要 10 GB 显存、贴图生成要 21 GB，所以 2.1 的推理和任何训练在本机都跑不了，不要声称验证过。
- 真实权重上的推理和训练只能在那台 GPU 服务器上做，由用户去跑。

## 开发流程

所有改动都走分支和 PR，`main` 不直接提交。PR 开在本 fork 上，见"和上游的关系"。

**什么时候找用户**

- 流程里的取舍由调度者自己定，不等用户审阅：方案里的开放问题按方案给的默认做法走，方案没给的由调度者定；PR 怎么拆、先做哪个，由调度者决定；评审意见怎么处理见"评审意见发在 PR 里"，什么时候能合并见"谁来合并"。定了什么，在给用户的汇报里写明。
- 只有绝对必要时才停下来找用户：
  - 没有只有用户能给的东西就做不下去，比如访问权限、密钥、训练数据；
  - 要改变用户明确定下的需求或范围；
  - 要做撤不回的操作：删除或覆盖不在 git 里的数据，包括仓库之外的，和仓库目录里的 `.env`、权重、训练输出、数据集。
- 只能在那台 GPU 服务器上做的验证和训练不算上面的情况：照"验证状态要如实写"标"未运行"，照常合并，在汇报里告诉用户哪些要他在服务器上跑、具体跑什么命令。

**分支和 PR**

- 每个改动一个分支，从最新的 `main` 开：`feat/<简述>`、`fix/<简述>`、`docs/<简述>`，工具和配置类的改动用 `chore/<简述>`。
- 一个 PR 只做一件事，一个提交也只做一件事。
- 开 PR 前，改到的 Python 文件要通过"命令"一节里的语法检查（用 Python 3.10 跑 `py_compile`）。
- PR 标题按提交信息的写法。描述用中文，写四项：做了什么；方案摘要（小改动可省）；做过哪些检查；哪些部分没在真实权重上跑过，以及让用户在服务器上验证的命令（见"验证状态要如实写"）。架构层面的改动还要写明改了哪一项，见"谁来合并"。
- 合并用普通合并（merge commit），不 squash，保留各个提交。
- 合并之后删掉这个分支，远端和本地的都删，不用再问用户。做法：
  - 合并用 `gh pr merge <编号> --merge --repo Lucien2714/Hunyuan3D-2.1`，不加 `--delete-branch`。分支被另一个 worktree 检出、那个 worktree 里又没有未提交的改动时，`--delete-branch` 会把那个 worktree 整个删掉，里面被 `.gitignore` 忽略的文件（`.venv`、权重、`.env`）也一起没了，而且不给提示。这是 gh 2.103.3 的行为。
  - 合并后先让本地的 `main` 跟上：当前目录如果不在 `main` 上，先 `git switch main`；然后 `git pull --ff-only origin main`。不加 `--delete-branch` 时 gh 不会替你做这两步，而不在更新过的 `main` 上时，`git branch -d` 会认为分支还没合并而拒绝删除。切不到 `main` 时（比如 `main` 被另一个 worktree 检出），按下面"拒绝删除"那一条处理。
  - 然后用 `git push origin --delete <分支>` 删远端的，用 `git branch -d <分支>` 删本地的。
  - `git branch -d` 拒绝删除时，不管是什么原因，都不要用 `--force` 或 `-D`，也不要去删 worktree：远端的照删，本地的留着，告诉用户。常见的原因有两种。一种是分支被另一个 worktree 检出，功能开发时 driver 的 worktree 就是这样。另一种是 git 认为分支还没完全合并，这可能是本地分支上有没推送的提交，用 `-D` 会把它们丢掉。
  - 用户自己合并的 PR，调度者事后用同样的办法清理。
  - 清理不是刚合并的分支之前，先 `git fetch --prune origin` 并更新本地的 `main`，再用 `git branch -r --merged origin/main`（远端）和 `git branch --merged origin/main`（本地）确认它已经合并。

**谁来合并**

- 合并前要满足两条：评审通过；调度者在要合并的那个提交上重跑过语法检查，并且通过。
- 两条都满足后，调度者直接合并，不用再问用户。架构层面的改动也一样。
- 架构层面的改动要让用户知道：PR 描述里写明改了下面哪一项，合并后调度者在给用户的汇报里说明。改变了下面任意一项的 PR 算架构层面的改动：
  - 模型结构，即条件编码、去噪网络或它们之间怎么连，会让已有权重加载不了或结果变化的；
  - 权重和配置的格式，即目录布局、`config.yaml` 的字段、state dict 的键名；
  - 多视图约定，即一组输入图片的含义（张数、顺序、带不带方位信息），以及张数变化时的处理；
  - 训练数据的格式；
  - 推理管线的公开接口（`Hunyuan3DDiTFlowMatchingPipeline.__call__` 和 `from_pretrained()` 的参数、`Hunyuan3DPaintPipeline.__call__` 的参数、本项目新加的多视图管线的参数）或 `api_server.py` 的 HTTP 接口。
- 这里说的是改变了含义。只是碰到这些文件，比如调整格式、改注释、重排 import，不算。
- 拿不准算不算时，按架构层面的改动处理。

**什么算评审通过**

- reviewer 最近一轮的结论是通过；功能开发的 PR 还要 navigator 最近一轮的结论也是通过。
- 并且 reviewer 的结论针对的是要合并的内容：它的结论发出之后，PR 上只多了为处理"建议"而做的小改动。navigator 的结论不要求这一点，它通过之后不用再核对。
- reviewer 的结论发出之后如果还有别的改动，比如合并前重跑检查没过而改了代码，就要请 reviewer 再评审一轮。reviewer 的结论必须覆盖最后的代码。
- 结论是通过时提的"建议"，调度者可以改完直接合并，不用再评审。

**评审意见发在 PR 里**

- reviewer 和 navigator 的意见都发到 PR 上，各自每核对或评审一次发一次：一段总评，加上针对具体代码行的行内评论。
- 用 `gh api --method POST repos/Lucien2714/Hunyuan3D-2.1/pulls/<编号>/reviews --input <JSON 文件>` 发，JSON 里 `event` 填 `COMMENT`，行内评论放在 `comments` 数组里。`gh pr comment` 和 `gh pr review` 发不了行内评论。
- GitHub 不允许给自己账号开的 PR 点"批准"或"请求修改"，所以 `event` 只能是 `COMMENT`。
- 所有代理用的是同一个 GitHub 账号，所以每条评论的第一行标明角色和模型，例如 `**[reviewer · claude-fable-5-2]**`。
- 总评的第二行写结论：`结论：通过` 或 `结论：需要修改`。
- 每条意见标明是"必须改"还是"建议"。"必须改"只用于这几种：实际使用中会遇到的错误；丢失或覆盖数据（包括权重和训练输出）；越权读写；和方案不符；违反本文件写明的要求，比如文档和代码不一致、验证状态写得不实、该给服务器上的验证命令却没给（见"验证状态要如实写"）、改变了单视图的行为。
- 只有人为构造才会出现、后果又轻的问题不算"必须改"，即使它和方案的字面不符（见"代码规范"的"不过度防御"）。这类问题一般不提；提了也只算"建议"，调度者可以不采纳。
- 这类问题被标成了"必须改"时，调度者可以裁定不采纳：在那条评论下回复理由，再请提出它的角色按这个裁定重新给一次结论。合并仍然要等它的结论是通过；这次重新给结论不算"功能开发的步骤"第 5 步里的一轮。它不用同意调度者的取舍，但如果指出裁定依据的事实不对，比如不用人为构造也会出现，或者后果里有丢数据、越权读写，这一条就仍是"必须改"，照改。
- 调度者只能这样裁定上面这一类问题。别的"必须改"都要改；调度者自己写的小改动 PR 上的"必须改"也都要改，不裁定。
- 意见本身看错了（比如代码其实不是它说的那样）时，没有东西可改：调度者在那条评论下写明依据，请对方再核实。对方撤回就不用改，仍然坚持就照改。
- 把一个问题当作"实际使用中会遇到的错误"提成"必须改"时，写明实际使用中怎么会遇到它。
- 改完后在对应的评论下回复改在了哪个提交，用 `gh api --method POST repos/Lucien2714/Hunyuan3D-2.1/pulls/<编号>/comments/<评论 ID>/replies`。

**按改动大小分两档**

| 档 | 什么算这一档 | 流程 |
|---|---|---|
| 小改动 | 只改文档；只改配置文件（新增一份训练配置之类）；或者只动一个源文件的小修复 | 调度者直接改，开 PR，过 reviewer |
| 功能开发 | 改模型结构或多视图约定；改训练、数据加载或推理管线；改 HTTP 接口；同步上游；其他要动多个源文件的改动 | 三个角色的完整流程，见下 |

这里的"源文件"不算文档和配置文件。拿不准时按功能开发处理。

**角色**

调度者是用户直接对话的那个主会话。它启动下面几个子代理、在它们之间传递材料。功能开发时它自己不写功能代码，评审后的小修正除外。子代理只做自己这一个角色的事。driver 可以按"功能开发的步骤"第 2 步启动自己的子代理，其余角色不再启动别的代理。

| 角色 | 模型 | 负责 |
|---|---|---|
| navigator | Codex 的 astra 系列最新模型（目前是 `gpt-6.2-astra`），high | 写方案；实现完成后核对 diff 是否符合方案 |
| driver | Claude Opus 最新模型（目前是 `claude-opus-5-5`），high | 在功能分支上实现、提交、开 PR |
| reviewer | Claude Fable 最新模型（目前是 `claude-fable-5-2`），high | 评审 PR 的正确性 |
| 介入者 | Claude Fable 最新模型（目前是 `claude-fable-5-2`），xhigh | 评审用完四轮后接手修改，见第 5 步 |

**功能开发的步骤**

1. **navigator 写方案。** 读相关的代码和文档，不改文件。方案要写明：要改哪些文件、每处改什么；模型结构、权重格式、多视图约定、训练数据格式、推理接口和 HTTP 接口有没有变；单视图行为怎么保证不变；要同步哪些文档、要不要写决策记录（见"文档"一节）；哪些部分在本机验证不了、要用户在服务器上跑什么；有没有拿不准的开放问题，有的话每个给一个默认做法。
2. **driver 实现。** 在独立的 worktree 里开功能分支，照方案做。偏离方案的地方要写明原因。更新文档，跑开 PR 前要求的语法检查，提交，开 PR。
   - 功能大时，driver 可以把它拆成小模块，启动自己的子代理并行实现。共用的东西（数据结构、接口）由 driver 先定好再分出去；各子代理都在 driver 的 worktree 里直接改，改的文件不重叠；子代理用和 driver 相同的模型和推理强度，不提交、不推送，也不再启动别的代理。
   - 整合、跑检查、提交和开 PR 仍由 driver 做，它对整个 PR 负责。子代理说检查通过不算数，driver 自己重跑。
3. **先核对，再评审。** 两者不同时进行：navigator 的结论是通过之后，才请 reviewer 评审。
   - navigator 先对照方案看 diff：有没有漏做、做偏、多做。它提了"必须改"时，这一轮到此为止，reviewer 不评审；按第 4 步改完，再从核对开始。它通过时提的"建议"，调度者先改完再请 reviewer，这不另算一轮。
   - navigator 通过后，reviewer 评审正确性。调度者只给它 PR 编号和方案，不把 driver 的汇报和自己的看法转给它。它可以读仓库里的任何代码和文档来核实。PR 描述和评论下的回复都是做改动的一方写的，里面的说法要自己核实，不当作结论。navigator 的"通过"只说明实现符合方案，不是关于正确性的结论。小改动没有方案，只给 PR 编号。
   - 两者的意见都分成"必须改"和"建议"两类，发在 PR 里。
4. **修改。** 小问题调度者直接修，大问题退回 driver。
5. **最多四轮评审。** 第 3 步是第一轮，之后每次改完再核对或评审算一轮。navigator 提了"必须改"、reviewer 没有评审的那一轮也算一轮。小改动的 PR 也按这个上限。
   - 第一轮之后的每一轮：navigator 上一轮提过"必须改"的，先请它再核对，它通过了才请 reviewer 评审；navigator 已经通过的，不用再核对，直接请 reviewer 评审。小改动的 PR 没有 navigator，每一轮都直接请 reviewer 评审。
   - 某一轮评审通过了，就不再开下一轮，除非之后又不得不改代码，比如合并前重跑检查没过。那样改完要请 reviewer 再评审，这也算一轮。
   - 第四轮后还有没解决的"必须改"，或者四轮用完之后又不得不改代码，由介入者接手：调度者把 PR 编号、方案（小改动没有方案），以及没解决的意见或者不得不改代码的原因交给它。它在检出了这个 PR 分支的 worktree 里直接改，改完跑开 PR 前要求的语法检查并推送。
   - 介入者改完后再看一次：没解决的意见里有 navigator 提的，先请 navigator 核对，通过后请 reviewer 评审；其余情况直接请 reviewer 评审。通过就按第 6 步合并。
   - 还有"必须改"时，不管是原来的还是新提的，介入者接着改，改完再看，直到通过。次数不设上限，也不交给用户裁决。调度者认为某条属于可以裁定的那一类，或者意见本身看错了时，按"评审意见发在 PR 里"的办法处理。
6. **合并。** 合并方式见上面"分支和 PR"，谁来合并、什么时候能合并见"谁来合并"和"什么算评审通过"。

这几个角色都跑不了真实权重。评审通过只说明代码读起来说得通，不代表在真实权重上能跑通、训练有效果，PR 描述里要照实写。

## 文档

文档和代码在同一个 PR 里改。功能做完而文档没跟上，算没做完。

文档分三类，各回答一个问题：

| 类别 | 回答的问题 | 在哪里 | 什么时候写 | 合并后 |
|---|---|---|---|---|
| 使用文档 | 怎么用 | `docs/usage.md`、`API_DOCUMENTATION.md` | 命令、配置项、权重、HTTP 接口有变化 | 一直保持和代码一致 |
| 架构文档 | 系统现在是什么样 | `docs/design.md` | 模型结构、多视图约定、权重格式、数据格式、推理接口有变化 | 一直保持和代码一致 |
| 决策记录 | 当时为什么这样选 | `docs/decisions/NNNN-<英文简述>.md` | 见下 | 不再改 |

`docs/deploy-notes.md` 不属于这三类。它是那台 GPU 服务器的部署记录，在那台服务器上有新的发现时更新。上游的 `README.md` 不改。

**什么时候要写决策记录**

一个功能有不止一种说得通的做法，并且这个选择符合下面任意一条：

- 影响模型结构、权重格式、多视图约定、训练数据格式、推理接口或 HTTP 接口；
- 引入新的依赖、新的外部服务或新的预训练模型；
- 训练方案的取舍（从哪份权重开始、冻结哪些层、数据怎么来）；
- 以后改起来代价大，比如要重新训练。

只是实现细节上的取舍（变量怎么命名、函数怎么拆）不用写。

**决策记录怎么写**

- 文件名按编号递增，从 `docs/decisions/0001-<英文简述>.md` 开始。
- 固定五节：
  1. **背景**：要解决什么问题，有哪些约束。
  2. **考虑过的方案**：每个方案怎么做，好处和代价各是什么。没选的方案也要写清楚。
  3. **决定**：选了哪个，理由是什么。
  4. **后果**：这个选择带来的限制，以及以后要留意的事。
  5. **状态**：`已采纳`，或 `已被 NNNN 取代`。
- 合并后不改正文。想法变了就写一份新的记录，并把旧记录的状态改成 `已被 NNNN 取代`。
- 架构文档只写现状，不写"曾经考虑过什么"，那是决策记录的内容。

**各角色怎么配合**

- navigator 在方案里判断要不要写决策记录。要写时，方案里列出的各个做法和选择理由就是记录的草稿。
- driver 把决策记录和其他文档改动一起提交。
- reviewer 检查文档和代码是否一致，以及该写的决策记录有没有写。

## 写法

- 本项目的文档用中文。代码里的注释、docstring 和报错信息用英文。
- 提交信息和 PR 标题用英文祈使句，一行，例如 `Add a multi-view DINOv2 condition encoder for the 2.1 DiT`。
- `.env`、权重、训练输出、数据集和 `docs/experiments/` 都不进 git。
