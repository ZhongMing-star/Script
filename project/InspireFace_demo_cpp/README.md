使用 InspireFace C++SDK, 封装三个类：
1.  InspireFaceGlobal 全局 SDK 管理器，负责初始化、销毁 SDK。
2.  InspireFaceDetector 人脸检测器，负责检测人脸。
3.  InspireFaceRecognizer 人脸识别器，负责识别人脸。

InspireFaceGlobal 设置为单例模式，确保全局只有一个实例。
InspireFaceDetector、 InspireFaceRecognizer 读取全局的 InspireFaceGlobal 中的配置参数。且不需要主动管理 session。
InspireFaceDetector 对外一个接口 detect，负责检测人脸。 只需要传入 img 即可，返回检测结果。
InspireFaceRecognizer 对外一个接口 recognize，负责识别人脸。 只需要传入 原图和检测结果 即可，返回人脸特征向量。