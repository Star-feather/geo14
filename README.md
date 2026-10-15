# geo_14_ 运行说明与数据泄露标注

2026-10-15，本机 RTX 4060 Laptop 8GB 实测。主入口为 `main_h36m.py`，不要直接运行 model2/train.py 或 model2/test.py。

**按用户要求：保留原始数据泄露，不修正 gt.reshape、不替换几何解码、不改变原始验证/测试划分。只在原地赋值相关位置增加 clone，修复训练反向传播错误；另补充运行参数、长度同步和测试日志输出支持。本文中的 MPJPE 仍受数据泄露影响，不能作为严格历史输入预测的性能。**

## 1. 环境与数据

PowerShell 初始化，后续命令均在此基础上运行：

```powershell
Set-Location D:\codex\geo\geo_14_
$py = 'D:/software/anaconda/envs/pytorch/python.exe'
$env:PYTHONPATH = "$PWD/vendor"
$env:PYTHONIOENCODING = 'utf-8'
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

本机使用 Python 3.12、PyTorch 2.6.0+cu118、NumPy 2.1.2、SciPy 1.15.2、pandas 2.2.3、matplotlib 3.10.1、einops 0.8.2、tensorboard 2.19.0。默认 python 是 CPU 版，因此使用上面的明确解释器路径。GPU 环境缺少的 easydict 1.13 已从本机其他环境复制到 vendor/easydict；设置 PYTHONPATH 后可用，未修改全局环境。

迁移环境需先准备适合机器的 CUDA PyTorch，再安装 numpy scipy pandas matplotlib einops tensorboard easydict。原 requirements.txt/environment.yml 是旧完整环境导出，不是本次运行环境的锁文件。

真实 H36M 数据根目录：`D:/codex/datasets/h3.6m`。下有 S1/S5/S6/S7/S8/S9/S11，各主体包含 walking_1.txt、walking_2.txt 等 15 动作×2序列，共 210 文件。

预处理：每隔 2 帧采样、清零全局平移与旋转、前向运动学得到 32 关节坐标，从中选择 22 关节（66 维）进行几何编码。降采样后每帧 40 ms，25 帧是 1000 ms。

### 新建独立环境（另一台机器也可用）

已提供 `environment-run.yml` 和 `requirements-run.txt`。Python 实测版本为 **3.12.9**；PyTorch wheel 自带 CUDA 11.8 运行时，本机 NVIDIA 驱动572.83。无需为本项目另外编译CUDA扩展。程序有模块级 `.cuda()` 调用，当前运行方式要求可用NVIDIA GPU。

```powershell
Set-Location D:\codex\geo\geo_14_
conda env create -f environment-run.yml
conda activate geo14
python -c "import torch, scipy, pandas, matplotlib, einops, tensorboard, easydict; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
$py = (Get-Command python).Source
$env:PYTHONIOENCODING = 'utf-8'
```

也可逐步安装：

```powershell
conda create -n geo14 python=3.12.9 pip -y
conda activate geo14
python -m pip install -r requirements-run.txt
$py = (Get-Command python).Source
```

两种方式任选其一。新环境已安装easydict，不需要vendor的PYTHONPATH。所有后续命令使用 `$py`，因此可以选择现有已验证环境，或切换为新建的geo14环境。

依赖文件锁定的是项目直接依赖，传递依赖由pip解析，不是完整跨平台锁文件。本轮没有额外新建/联网安装一个环境；实际训练测试运行于上述既有pytorch环境。机器和包版本核验记录在 `audit/environment.json`。

## 2. 输入多少、输出多少

| 参数 | 含义 | 默认与限制 |
|---|---|---|
| `--input_n` | 历史输入帧数 | 10；当前架构固定为10 |
| `--kernel_size` | model1 历史核长度 | 10；必须与 input_n 一致 |
| `--output_n` | 每次前向预测未来帧数 | 25；改变后需要重新训练 |
| `--dct_n` | 每次内部历史+预测总长 | 必须为 input_n+output_n，默认35 |
| `--test_length` | 测试总共预测未来帧数 | 默认25；必须是 output_n 的正整数倍 |
| `--batch_size` | 训练批大小 | 16 |
| `--test_batch_size` | 验证/测试批大小 | 16 |
| `--epoch` | 总训练轮数 | 50；恢复训练时也是目标总轮数 |
| `--skip_rate` | 滑窗起点步长 | 默认1；同时影响训练/验证/测试 |
| `--max_batches` | 每个加载器最多跑多少批 | 0为全部，正数用于流程检查 |
| `--actions walking eating` | 限定动作列表 | 不指定则15动作全部使用 |
| `--is_eval` | 只测试 ckpt_best | 不加则训练，已修正原来恒为True的默认值 |
| `--is_load` | 从 ckpt_last 恢复训练 | 与原训练模型参数保持一致 |
| `--ckpt` | 检查点根目录 | 自动追加实验名子目录 |
| `--result_dir` | 独立测试CSV输出目录 | 建议每次使用新目录 |

训练期间要求 test_length=output_n；单独测试允许滚动预测。例如10→25为 `--input_n 10 --output_n 25 --dct_n 35 --test_length 25`；10→10为 `--input_n 10 --output_n 10 --dct_n 20 --test_length 10`。

内部模型输出历史+未来，默认 `[B,35,66]`；未来指标只计算后25帧。**不能只改 input_n 来支持任意历史长度**：模型硬性要求 kernel_size=10，已增加清晰的参数检查。改变 output_n/dct_n 会改变网络权重形状，不能加载原来的10→25检查点。不同长度均已同步 model2 配置及 DCT 缓存。

## 3. 命令行运行

### 3.1 快速跑通训练：10→25，1轮稀疏滑窗

```powershell
& $py -u main_h36m.py --data_dir D:/codex/datasets/h3.6m --ckpt audit/preserved_train --input_n 10 --output_n 25 --dct_n 35 --test_length 25 --epoch 1 --skip_rate 1000 --batch_size 16 --test_batch_size 16
```

本轮最终版本按此设置实际执行：全部15个动作、真实数据、从头训练1轮、验证、测试和保存检查点。是运行检查，不是全量滑窗或收敛训练。日志：`audit/preserved_train.log`。skip_rate=1000训练249样本，共16次参数更新。

### 3.2 常规训练：10→25，全部滑窗50轮

```powershell
& $py -u main_h36m.py --data_dir D:/codex/datasets/h3.6m --ckpt experiments/geo14_10to25 --input_n 10 --output_n 25 --dct_n 35 --test_length 25 --epoch 50 --skip_rate 1 --batch_size 16 --test_batch_size 16
```

这是完整训练命令，本轮没有执行50轮全量训练。由于按要求保留泄露，其结果仍不代表无泄露实验。

恢复同一实验训练到第60轮：

```powershell
& $py -u main_h36m.py --data_dir D:/codex/datasets/h3.6m --ckpt experiments/geo14_10to25 --is_load --input_n 10 --output_n 25 --dct_n 35 --test_length 25 --epoch 60 --skip_rate 1 --batch_size 16 --test_batch_size 16
```

### 3.3 加载本轮训练检查点测试：10→25

```powershell
& $py -u main_h36m.py --is_eval --data_dir D:/codex/datasets/h3.6m --ckpt audit/preserved_train --input_n 10 --output_n 25 --dct_n 35 --test_length 25 --skip_rate 1000 --result_dir audit/preserved_reloaded
```

正式实验训练结束后，把 ckpt 换成 experiments/geo14_10to25，skip_rate 改为1，并选择新的 result_dir 即可。ckpt 传根目录，程序自己拼接子目录。

### 3.4 测试原始epoch44检查点：10→25

```powershell
& $py -u main_h36m.py --is_eval --data_dir D:/codex/datasets/h3.6m --ckpt checkpoint --input_n 10 --output_n 25 --dct_n 35 --test_length 25 --skip_rate 100 --test_batch_size 16 --result_dir audit/original_test
```

原始reshape模式已对全部15动作完成测试，先前记录在 audit/legacy_eval.log 和 audit/legacy_eval.csv。以上命令将结果写入新目录，保留原检查点目录中的历史CSV。

### 3.5 改为10→10，单独训练和测试

```powershell
& $py -u main_h36m.py --data_dir D:/codex/datasets/h3.6m --ckpt audit/preserved_length10 --actions walking --input_n 10 --output_n 10 --dct_n 20 --test_length 10 --epoch 1 --skip_rate 1000

& $py -u main_h36m.py --is_eval --data_dir D:/codex/datasets/h3.6m --ckpt audit/preserved_length10 --actions walking --input_n 10 --output_n 10 --dct_n 20 --test_length 10 --skip_rate 1000 --result_dir audit/preserved_length10_test
```

### 3.6 输入10帧，分4次滚动预测未来100帧

```powershell
& $py -u main_h36m.py --is_eval --data_dir D:/codex/datasets/h3.6m --ckpt audit/preserved_train --actions walking --input_n 10 --output_n 25 --dct_n 35 --test_length 100 --skip_rate 1000 --result_dir audit/preserved_rollout100
```

output_n仍为25、dct_n仍为35，每次取预测的最后10帧作为下一次输入。不要把dct_n设为110。本模式保留原始reshape，因此第一步会读取测试序列中的未来真值，不能称为严格历史输入的100帧预测。

## 4. 数据泄露在哪里（按要求保留）

### LEAK-01：model/model_h36m.py 的 gt.reshape

位置：`MultiStageModel_4.forward()` 的 `gt = src.clone()` 和残差 `residual_input` 中的：

```python
gt.reshape(bs, -1, self.node_n, dct_n)[:, :, :, context_start:update_idx]
```

main_h36m.py 测试路径通过 `input_data_geo1_ = input_data_geo1.clone()` 向model1传入完整真实序列。默认35帧包含历史0–9和未来10–34；主干 input_gcn1 虽用 idx 截取历史并复制末帧，残差支路仍从完整 gt 取值。

原始gt布局是 `[B,T,66]`，也就是 `[B,T,22,3]`。reshape不会交换轴，把它直接解释成 `[B,3,22,T]` 后，最后一轴不是真正的时间轴。“只截取末轴的历史窗口”仍会读到未来原始帧。

对35帧布局逐元素追踪：4个上下文窗口共取1056个元素，其中752个来自未来，源时间覆盖0–34。该处已在代码添加 LEAK-01 注释，**计算表达式保留不变**。

训练原本只传10帧，因此这里训练时没有未来输入，但轴顺序仍错误；测试传入完整序列才引入未来。forward中的局部dct_n取实际src长度，训练时为10，原始reshape本身能够执行，训练失败并非必然的reshape维度错误。

复现依赖关系：

```powershell
& $py audit_leakage.py
```

固定历史10帧，只将未来25帧所有元素加10，原epoch44权重的输出最大变化为 **104.2784958**。诊断脚本还在自己的进程内临时替换历史参考表达式做对照，变化为0；这个替换只存在于诊断脚本内，**没有修改生产模型的数据泄露路径**。详细结果在 audit/leakage_check.json。随机输入只用于依赖检查，不是性能评估。

### LEAK-02 / LEAK-03：utils/geometric.py 的未来骨长和符号

main_h36m.py 对完整真实序列调用 encode，缓存：

```python
self.bones = torch.norm(output_t, dim=-1)
self.refer = torch.sum(fix_vectors * output_t, dim=-1)
```

decode用未来bones计算两个候选骨向量的幅度，用 `torch.sign(self.refer)` 选正负分支。未来真实符号不属于仅历史输入；model1/model2各自解码后的笛卡尔指标直接受它辅助。

final融合使用b11/b12/b21/b22候选，不直接使用decode返回的符号掩码，因此不能把final描述成直接读取未来refer；但候选幅度仍使用完整真实序列的bones。该FK模板的多数骨长接近常量，访问未来骨长的影响大小仍需逐骨测量。

build_framework还读取完整序列的leg_right/leg_left；清零全局运动后这些根部关节可能固定，不能未经核查就把它们说成具有同等影响的动态泄露。这些访问均保留。代码中已标注LEAK-02/LEAK-03。

### LEAK-04：验证集与测试集都用S5

utils/h36motion3d.py 实际映射：split=0为S1/S6/S7/S8/S9，split=1为S11，split=2为S5。主入口原来验证和测试都用split=2，再用验证指标选择best，因此S5参与选模型。

**按要求维持原始split=2验证及测试，没有换成S11。** main_h36m.py已添加LEAK-04注释。本轮运行只是复现并跑通原方法，不能声称测试集完全独立。

## 6. 流程、结果和日志怎么读

数据 → 两路GEOMETRIC编码（固定向量分别沿y轴与x轴）→ model1多级GCN与model2 siMLPe分别预测 → 几何解码和候选融合 → 骨架恢复 → 32关节MPJPE。

实际反传损失为 `loss_geo1 + loss_geo2`。optimizer1包含两个模型参数并执行step；optimizer2创建且保存，但当前不执行step。融合后的笛卡尔loss只是统计，不参与本轮反传。

默认检查点路径：

```text
<ckpt根目录>/main_h36m_all_in10_out25_ks10_dctn35_dropout_0.3_lr_0.005_d_model_16_e_6_d_6_idctbeofre/
```

all来自test_sample_num，不表示skip_rate一定为1；skip_rate不进入目录名，改设置时建议换ckpt根目录。

- ckpt_best.pth.tar：按原S5验证model1误差选出的检查点。
- ckpt_last.pth.tar：最后轮次，保存两个模型和优化器状态。
- test_all_eval.csv：各动作及动作宏平均avg；每帧一列，单位mm。100帧时输出到#4000ms。
- test.csv：原格式把model1/model2/final依次写入，列结构不统一；优先读取test_all_eval.csv。
- audit/preserved_train.log：最终保留泄露版本训练日志。

终端“测试误差”是最后动作的最后时间点，不是所有动作平均；验证段仍沿用“训练误差”文字，需结合上下文查看。正式统计应读CSV的avg行。

先前原epoch44权重、15动作、skip_rate=100的泄露模式测试：final动作宏平均80ms=22.539mm、400ms=96.152mm、1000ms=84.551mm。它只代表原方法含泄露的结果。

`audit/train`、`audit/fixed_eval`、`audit/length10`、`audit/rollout100` 是本次早期诊断产生的历史参考，部分使用了临时修正的reshape或S11验证，**不是最终保留泄露版本**；最终运行以preserved前缀文件为准。没有完成50轮全量收敛训练。

另有IDCT后及最终输出前的直接reshape可能存在轴语义问题，本轮保留，不把它们与未来数据泄露混为一谈。

### 最终保留泄露版本的实测摘要

完成1轮、16次优化器更新。S5验证model1=228.846mm、model2=107.993mm。S5全部15动作、skip_rate=1000的final宏平均：80ms=444.538mm、400ms=482.078mm、1000ms=372.986mm。独立重新加载测试与训练末尾测试结果完全一致；100帧测试输出100列时间指标且数值有限。10→10也完成训练/验证/测试及保存检查点。见 `audit/preserved_summary.json`。这些仍是含泄露的稀疏采样流程验证结果。
