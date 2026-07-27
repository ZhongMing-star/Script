# camera_calibration_cpp 使用说明书

`camera_calibration_cpp` 是一个基于 ROS 2、OpenCV 和 C++ 的单目相机标定节点。节点订阅相机原始图像，自动检测棋盘格内角点，收集足够数量的有效图像后计算相机内参和畸变系数，并将结果保存为 ROS 相机标定 YAML 文件。

## 1. 功能概览

- 订阅 `sensor_msgs/msg/Image` 图像话题
- 自动检测棋盘格内角点并按时间间隔采样
- 将每张有效样本以带角点标注的 PNG 图像保存到磁盘
- 弹窗显示最新捕获的有效图像，下一次捕获时自动替换
- 通过 ROS 2 服务控制开始、停止、清空和执行标定
- 输出相机矩阵、畸变系数、校正矩阵和投影矩阵
- 自动创建输出文件的父目录

> 本程序只进行单目相机内参标定，不包含双目标定、手眼标定或标定结果的实时去畸变显示。

## 2. 环境与依赖

推荐环境：

- Ubuntu 22.04 + ROS 2 Humble，或其他具备相同依赖的 ROS 2 发行版
- C++17 编译器
- OpenCV
- ROS 2 软件包：`ament_cmake`、`rclcpp`、`sensor_msgs`、`std_srvs`、`cv_bridge`

在 Ubuntu/ROS 2 Humble 中可安装依赖：

```bash
sudo apt update
sudo apt install ros-humble-cv-bridge ros-humble-sensor-msgs ros-humble-std-srvs
```

如果使用的不是 Humble，请将命令中的 `humble` 替换为实际 ROS 2 发行版名称。也可以在工作空间根目录使用 `rosdep`：

```bash
rosdep install --from-paths src --ignore-src -r -y
```

## 3. 编译

假设本软件包位于工作空间 `~/ros2_cpp_ws/src/camera_calibration_cpp`：

```bash
source /opt/ros/humble/setup.bash
cd ~/ros2_cpp_ws
colcon build --packages-select camera_calibration_cpp
source install/setup.bash
```

每次新开终端后，都需要加载 ROS 2 和当前工作空间环境：

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_cpp_ws/install/setup.bash
```

## 4. 准备标定板

默认配置使用：

- 横向内角点数：12
- 纵向内角点数：9
- 相邻内角点的实际距离：0.04 m，即 40 mm

这里的宽和高是棋盘格的**内角点数量**，不是黑白方格数量。例如，`12 x 9` 内角点对应 `13 x 10` 个方格。`square_size` 是相邻角点之间的距离，也就是单个方格的边长；推荐使用米。

标定板应固定在平整、刚性的平面上。打印时关闭“适应页面”等缩放选项，并用尺子测量打印后的实际方格边长。

### 棋盘与镜头的位置关系

- 标定过程中固定相机、镜头焦距、变焦、对焦和图像分辨率；标定后改变其中任一项通常需要重新标定。
- 棋盘应完整进入画面且角点清晰。不要贴得过近导致边缘出画，也不要离得过远导致每个方格只占少量像素。
- 让棋盘在多数图像中占画面约 `1/3` 至 `2/3`，同时采集近、中、远距离，距离范围应覆盖相机的实际工作距离。
- 分别把棋盘移动到画面中心、四边和四角，尤其要覆盖镜头畸变更明显的边缘区域。
- 绕棋盘的水平轴和垂直轴做不同方向的倾斜，建议包含约 `15°` 至 `45°` 的透视角度；同时保留少量接近正对镜头的图像。
- 不要只让棋盘始终与成像平面平行，也不要只在画面中心前后移动，否则参数约束不足。
- 避免反光、过曝、运动模糊和严重阴影；采集时保持棋盘平整，不要弯折。
- 对自动对焦镜头，先在实际工作距离完成对焦后锁定；无法锁定时应避免采集期间焦点频繁变化。

## 5. 快速开始

先启动相机驱动并确认图像话题有数据，然后按以下步骤操作。

### 5.1 启动节点

```bash
ros2 run camera_calibration_cpp camera_calibration_node --ros-args \
  -p image_topic:=/hec/sensor/rgb_camera/head \
  -p board_width:=11 \
  -p board_height:=8 \
  -p square_size:=0.025 \
  -p required_frames:=20 \
  -p capture_interval:=0.5 \
  -p show_captured_image:=true \
  -p captured_images_dir:=$HOME/.ros/camera_calibration_images \
  -p output_file:=$HOME/.ros/camera_calibration.yaml
```

将棋盘参数改成实际内角点数量和实测方格边长。节点使用 `SensorDataQoS` 订阅图像。

### 5.2 采集图像

另开终端并调用：

```bash
ros2 service call /start_capture std_srvs/srv/Trigger '{}'
```

按照上一节的位置关系移动棋盘。每次有效捕获都会保存带角点标注的 PNG 并弹窗显示；下一次有效捕获时旧窗口关闭并显示新图。达到 `required_frames` 后自动停止采集。

### 5.3 执行标定

```bash
ros2 service call /calibrate std_srvs/srv/Trigger '{}'
```

程序至少需要 3 帧有效样本，实际建议采集 15 至 30 帧且姿态丰富。响应中的 `reprojection error` 越小通常越好，但仍应通过去畸变图像验证结果。

## 6. 参数说明

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `image_topic` | string | `/camera/image_raw` | 输入图像话题 |
| `board_width` | int | `12` | 棋盘格横向内角点数 |
| `board_height` | int | `9` | 棋盘格纵向内角点数 |
| `square_size` | double | `0.04` | 方格实际边长，推荐单位为米 |
| `required_frames` | int | `30` | 达到此样本数后自动停止采集 |
| `capture_interval` | double | `0.5` | 两次有效采样之间的最短间隔，单位为秒 |
| `output_file` | string | `camera_calibration.yaml` | 标定结果输出路径 |
| `captured_images_dir` | string | `captured_images` | 有效样本图像的保存目录；相对路径基于节点启动目录 |
| `show_captured_image` | bool | `true` | 是否弹窗显示最新捕获的带角点标注图像 |

可在节点运行时查看实际参数：

```bash
ros2 param list /camera_calibration_node
ros2 param get /camera_calibration_node image_topic
```

参数在节点启动时读取。为避免一次标定过程中配置发生变化，建议通过启动命令设置参数，不要依赖运行时修改。

## 7. 服务说明

所有服务均使用 `std_srvs/srv/Trigger`：

| 服务 | 作用 |
| --- | --- |
| `/start_capture` | 开始或继续采集有效棋盘格图像 |
| `/stop_capture` | 暂停采集，已保存样本不会丢失 |
| `/clear_samples` | 清空当前已采集的全部样本 |
| `/calibrate` | 使用当前样本执行标定并写入 YAML 文件 |

常用命令：

```bash
# 暂停采集
ros2 service call /stop_capture std_srvs/srv/Trigger '{}'

# 清空样本并重新开始
ros2 service call /clear_samples std_srvs/srv/Trigger '{}'
ros2 service call /start_capture std_srvs/srv/Trigger '{}'
```

如果节点被放入命名空间，服务名也会随命名空间变化。可通过以下命令确认实际名称：

```bash
ros2 service list
```

## 8. 输出文件

输出 YAML 包含：

- `image_width`、`image_height`：标定图像尺寸
- `camera_matrix`：3 x 3 相机内参矩阵
- `distortion_model`：固定为 `plumb_bob`
- `distortion_coefficients`：畸变系数
- `rectification_matrix`：单位矩阵
- `projection_matrix`：3 x 4 投影矩阵

默认相对路径 `camera_calibration.yaml` 相对于**启动节点时的当前工作目录**。为便于查找，建议使用绝对路径设置 `output_file`。

每次成功识别的样本会保存为带棋盘格角点标注的 PNG，文件名包含样本序号和消息时间戳。只有图像成功写盘后，该帧才会计入标定样本。`/clear_samples` 只清空内存中的角点样本，不删除已保存图像，以保留采集证据；开始新一轮采集时建议指定新的 `captured_images_dir`。

默认会弹窗显示最新捕获的有效图像。捕获下一张有效图像时，节点先关闭旧窗口，再显示新图像。远程或无桌面环境中可以设置 `-p show_captured_image:=false`；若未检测到 `DISPLAY` 或 `WAYLAND_DISPLAY`，节点也会自动禁用预览并输出警告。

查看结果：

```bash
cat ~/.ros/camera_calibration.yaml
```

该格式可用于常见的 ROS 相机信息发布工具。使用前请确认具体相机驱动对标定文件 URI 和字段格式的要求。

## 9. 常见问题

### 一直没有采集日志

依次检查：

1. `image_topic` 是否与相机实际话题一致。
2. 图像消息是否为可转换到 `bgr8` 的常见编码。
3. 是否已经调用 `/start_capture`。
4. 棋盘格内角点数是否设置正确。
5. 棋盘格是否完整进入画面，并具有足够的清晰度和对比度。
6. 相机与节点的 ROS Domain ID 是否一致。

可使用以下命令辅助检查：

```bash
ros2 node info /camera_calibration_node
ros2 topic info -v /camera/image_raw
ros2 topic hz /camera/image_raw
```

### `/calibrate` 返回 `need at least 3 samples`

当前有效样本少于 3 帧。调用 `/start_capture` 并继续移动标定板采集。工程使用不建议只用最低数量，应尽量达到配置的 `required_frames`。

### 输出 `cannot open output file`

检查输出目录是否可写、路径是否合法、磁盘空间是否充足。建议指定用户有写权限的绝对路径，例如：

```bash
-p output_file:=$HOME/.ros/camera_calibration.yaml
```

### 重投影误差较大

常见原因包括标定板尺寸设置错误、打印缩放、角点模糊、样本姿态单一、棋盘格不平整或相机自动对焦造成焦距变化。清空样本并在固定焦距、稳定曝光条件下重新采集。

### 修改分辨率后还能沿用标定结果吗

不建议直接沿用。相机内参和图像尺寸相关；更改分辨率、裁剪方式、数字变焦、镜头焦距或对焦状态后，应重新标定，或在明确成像缩放关系的前提下正确换算内参。
