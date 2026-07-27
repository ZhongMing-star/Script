import shutil
import subprocess

import cv2


def find_device_users(device_path: str) -> str:
    fuser = shutil.which("fuser")
    if not fuser:
        return ""

    result = subprocess.run(
        [fuser, "-v", device_path],
        capture_output=True,
        text=True,
        check=False,
    )
    details = (result.stdout + result.stderr).strip()
    return details


def open_camera():
    candidates = [
        ("/dev/video0", cv2.CAP_V4L2),
        (0, cv2.CAP_V4L2),
        ("/dev/video0", cv2.CAP_ANY),
        (0, cv2.CAP_ANY),
    ]

    for source, backend in candidates:
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            return cap, source, backend
        cap.release()

    return None, None, None

# 打开摄像头，0=/dev/video0，多摄像头换1/2
cap, source, backend = open_camera()
# 可选：设置分辨率、帧率
if cap is not None:
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    pass

# 校验摄像头是否打开
if cap is None or not cap.isOpened():
    print("无法打开摄像头，请检查权限、设备编号或当前是否被其他进程占用")
    users = find_device_users("/dev/video0")
    if users:
        print("/dev/video0 当前占用情况:")
        print(users)
    exit()

print(f"已打开摄像头: source={source}, backend={backend}")

# 循环读取帧并显示
while True:
    ret, frame = cap.read()
    if not ret:
        print("读取画面失败，摄像头断开")
        break
    # 弹出窗口显示画面
    cv2.imshow("USB Camera", frame)
    # 按 q 键退出（1ms等待按键）
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()