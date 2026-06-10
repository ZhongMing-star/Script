# from insightface.app import FaceAnalysis
# from insightface.app.common import Face
# import cv2 

# model = FaceAnalysis(name="buffalo_sc").models['detection']
# img = cv2.imread("/mnt/d/Data/code/Script/project/demo_cpp/insightface_onnx_infer/resource/face.jpg")

# bboxes, kpss = model.detect(img)

# print(bboxes)


from model import SCRFD
import cv2
import time
model = SCRFD(model_file="/mnt/d/Data/code/Script/project/demo_cpp/merge_preprocess_to_onnx/model/det_500m_with_pre.onnx")
img = cv2.imread("/mnt/d/Data/code/Script/project/demo_cpp/insightface_onnx_infer/resource/face.jpg")

start = time.time()
bboxes, kpss = model.detect(img)
end = time.time()
print(f"detect time: {(end - start)*1000}")
# 将 bboxes 绘制在 img 上
for bbox in bboxes:
    x1, y1, x2, y2, _ = bbox
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(bboxes)
