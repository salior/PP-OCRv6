# PP-OCRv6 液晶数字摄像头识别示例

这个示例使用 `PaddleOCR` 的通用 OCR 管线。当前 `paddleocr 3.7.0` 默认使用 PP-OCRv6 模型，因此脚本会优先按 PP-OCRv6 初始化；如果本机安装的是旧版 API，会自动回退到兼容写法。

在 Windows CPU 环境下，脚本默认走 `onnxruntime` 引擎，避免部分 `paddle_static + oneDNN` 组合触发运行时报错。

## 1. 安装

建议先创建虚拟环境，再安装依赖。

```powershell
pip install paddlepaddle
pip install -r requirements.txt
```

如果你有 NVIDIA GPU，可以按 PaddlePaddle 官方文档换成对应 CUDA 版本的安装命令。

## 2. 运行

```powershell
python lcd_ocr_camera.py
```

常用参数：

```powershell
python lcd_ocr_camera.py --camera 0 --width 1280 --height 720 --interval 0.25
python lcd_ocr_camera.py --device gpu
python lcd_ocr_camera.py --roi-ratio 0.4 --scale 2.5
python lcd_ocr_camera.py --engine onnxruntime
```
d:\python\PP-OCRv6\.venv\Scripts\python.exe d:/python/PP-OCRv6/lcd_ocr_camera.py --engine onnxruntime --roi-ratio 0.4

## 3. 使用说明

- 默认只识别画面中央框，适合把液晶表盘对准框内，速度更快。
- 按 `c` 可以切换为全画面识别。
- 按 `q` 退出。

## 4. 提升识别率的建议

- 尽量让液晶数字占识别框宽度的 50% 以上。
- 避免反光，光线不均会影响阈值化效果。
- 如果是小数点不稳定，可以把摄像头拉近一些，并把 `--scale` 调到 `2.5` 或 `3.0`。
- 如果背景干扰大，可以把 `--roi-ratio` 调小，只框住数字区域。