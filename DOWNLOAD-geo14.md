# 下载与使用

完整项目位于 `geo14-github-ready.zip`，包含源代码、原始检查点、本次训练检查点、README、依赖环境和运行记录，保留原始目录结构；不包含H36M原始数据、IDE配置和Python缓存。

下载压缩包后解压到独立目录，在解压后的项目根目录按README配置环境并运行。H36M数据需要自行准备，命令里的 `--data_dir` 应替换为自己的数据目录。

```powershell
Expand-Archive ./geo14-github-ready.zip ./geo14-project
Set-Location ./geo14-project
conda env create -f environment-run.yml
conda activate geo14
python main_h36m.py --data_dir D:/your-data/h3.6m --ckpt experiments/my_run --input_n 10 --output_n 25 --dct_n 35 --test_length 25 --epoch 1 --skip_rate 1000
```

原始数据泄露按照项目要求保留，位置和影响见README；上述命令用于复现原方法，不代表无泄露预测。
