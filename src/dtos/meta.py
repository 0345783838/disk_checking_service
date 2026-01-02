from pydantic import BaseModel


class DataResponse(BaseModel):
    result: bool = False  # True/False
    error_code: str = None  # ["PASS", "ERROR_001", "ERROR_002", "ERROR_003",...]
    error_desc: str = None  # ["Kem thiếc đạt chuẩn", "Kem thiếc in lệch", "Model này chưa được setup",...]
    res_img: str = None  # base64 result encoded image


class DataDebugResponse(BaseModel):
    result: bool = False  # True/False
    detect_img: str = None  # base64 result encoded image
    segment_img: str = None  # base64 result encoded image
    final_img: str = None  # base64 result encoded image


class ErrorCode:
    PASS = ("PASS", "Kiểm tra OK")
    ABNORMAL = ("ERROR_001", "Khay đĩa bất thường")
    ERR_NUM_DISK = ("ERROR_002", "Số lượng phát hiện khe đĩa bất thường")
