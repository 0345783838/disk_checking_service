import onnxruntime as ort
import numpy as np
import cv2


class OnnxSegmentor:
    def __init__(self, onnx_path: str, conf_thres: float = 0.5, gpu_mode=0, num_threads=8):
        """
        ONNX Inference Engine cho bài toán segmentation.
        - onnx_path: đường dẫn file .onnx
        - conf_thres: ngưỡng nhị phân hóa mask (0–1)
        - gpu_mode: 1 = CUDA, 0 = CPU
        - num_threads: số luồng CPU
        """
        self.conf_thres = conf_thres
        self._initialize_model(onnx_path, gpu_mode, num_threads)

    def __call__(self, image: np.ndarray):
        return self.infer(image)

    def _initialize_model(self, path, gpu_mode=0, num_threads=8):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if gpu_mode:
            self.session = ort.InferenceSession(path, providers=["CUDAExecutionProvider"])
        else:
            so.intra_op_num_threads = num_threads
            so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

        self._get_input_details()
        self._get_output_details()

        # Warmup
        dummy = np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)
        for _ in range(3):
            _ = self.session.run(self.output_names, {self.input_name: dummy})

    def _get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def _get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [out.name for out in model_outputs]

    def preprocess(self, image: np.ndarray):
        """
        Chuẩn hóa ảnh đầu vào: resize, chuyển (H,W,C)->(1,C,H,W), scale [0,1].
        """
        self.orig_h, self.orig_w = image.shape[:2]
        image_resized = cv2.resize(image, (self.input_width, self.input_height))
        image_resized = image_resized.astype(np.float32) / 255.0
        image_resized = np.transpose(image_resized, (2, 0, 1))  # HWC -> CHW
        image_resized = np.expand_dims(image_resized, axis=0)   # (1, C, H, W)
        return image_resized

    def infer(self, image: np.ndarray):
        """
        Chạy inference và trả về (mask_resized, score)
        """
        x = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: x})[0]
        return self.postprocess(outputs)

    def postprocess(self, outputs: np.ndarray):
        """
        Hậu xử lý kết quả:
        - outputs: (1, 1, H, W)
        - Trả về (mask_resized, score)
          + mask_resized: uint8 (0/255)
          + score: float
              = 1.0 nếu không có vùng lỗi
              = mean(prob_map[mask_thresh]) nếu có vùng lỗi
        """
        # Lấy ra mask logit
        mask_logits = outputs[0, 0] if outputs.ndim == 4 else outputs[0]

        # sigmoid
        prob_map = 1 / (1 + np.exp(-mask_logits))

        # tạo mask nhị phân
        mask_thresh = prob_map > self.conf_thres

        if mask_thresh.sum() == 0:
            # Không có lỗi nào được phát hiện
            score = 1.0
        else:
            # Lấy độ tin cậy trung bình của các vùng được dự đoán là lỗi
            score = float(prob_map[mask_thresh].mean())

        # Tạo mask nhị phân (0/255) và resize về kích thước ảnh gốc
        mask_bin = (mask_thresh.astype(np.uint8)) * 255
        mask_resized = cv2.resize(mask_bin, (self.orig_w, self.orig_h), interpolation=cv2.INTER_NEAREST)

        return mask_resized, score

