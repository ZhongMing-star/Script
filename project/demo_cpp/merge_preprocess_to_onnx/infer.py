import cv2 
from model import SCRFD


def draw_bboxes(img, bboxes):
    # 将 bboxes 绘制在 img 上
    for bbox in bboxes:
        x1, y1, x2, y2, _ = bbox
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

def main():
    det_model = SCRFD(model_file="/mnt/d/Data/code/Script/project/demo_cpp/merge_preprocess_to_onnx/model/det_500m_with_prep.onnx")
    
    post_fix = 2
    img = cv2.imread("/mnt/d/Data/code/Script/project/demo_cpp/insightface_onnx_infer/resource/face.jpg")
    bboxes, kpss = det_model.detect(img)
    draw_bboxes(img, bboxes=bboxes)
    cv2.imwrite(f"1-{post_fix}.jpg", img)
    
    img = cv2.imread("/mnt/d/Data/code/Script/project/demo_cpp/insightface_onnx_infer/resource/face2.jpg")
    bboxes, kpss = det_model.detect(img)
    draw_bboxes(img, bboxes=bboxes)
    cv2.imwrite(f"2-{post_fix}.jpg", img)
    
    img = cv2.imread("/mnt/d/Data/code/Script/project/demo_cpp/insightface_onnx_infer/resource/face3.jpg")
    bboxes, kpss = det_model.detect(img)
    draw_bboxes(img, bboxes=bboxes)
    cv2.imwrite(f"3-{post_fix}.jpg", img)

if __name__ == "__main__":
    main()
