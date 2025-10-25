# 实时人脸检测演示应用

这是一个基于Python Flask和YOLOv8的Web应用，用于教学演示目的。它通过网页界面实时检测摄像头中的人脸。

## 功能特点

- 实时人脸检测
- 基于YOLOv8深度学习模型
- 简洁的Web界面
- 易于部署和运行
- 优化的错误处理和资源管理
- 可配置的参数设置

## 技术栈

- Python 3.x
- Flask (Web框架)
- Ultralytics YOLOv8 (目标检测)
- OpenCV (计算机视觉库)

## 安装与运行

1. 克隆或下载此仓库
2. 安装依赖项：
   ```
   pip install -r requirements.txt
   ```
3. 运行应用：
   ```
   python app.py
   ```
4. 打开浏览器访问 `http://localhost:5000`

## 配置选项

可以通过环境变量配置应用：

- `CAMERA_INDEX`：摄像头索引（默认：0）
- `MODEL_PATH`：模型文件路径（默认：yolov8n.pt）
- `HOST`：绑定的主机地址（默认：127.0.0.1）
- `PORT`：端口号（默认：5000）
- `DEBUG`：调试模式（默认：True）

示例：
```bash
PORT=8080 MODEL_PATH=yolov8s.pt python app.py
```

## 注意事项

- 应用需要访问摄像头权限
- 为了更好的人脸检测效果，可以使用专门训练的面部检测模型替换默认的YOLOv8模型
- 本项目仅供教学演示使用
- 应用在退出时会自动释放摄像头资源