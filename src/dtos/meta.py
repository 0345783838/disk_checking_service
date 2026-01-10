from pydantic import BaseModel


class DataResponse(BaseModel):
    Result: bool = False  # True/False
    ErrorCode: str = None  # ["PASS", "ERROR_001", "ERROR_002", "ERROR_003",...]
    ErrorDesc: str = None  # ["Khay đĩa OK", "Khay đĩa bất thường",...]
    ResImg: str = None  # base64 result encoded image


class DataDebugResponse(BaseModel):
    Result: bool = False  # True/False
    DetectImg: str = None  # base64 result encoded image
    SegmentImg: str = None  # base64 result encoded image
    FinalImg: str = None  # base64 result encoded image


class ErrorCode:
    PASS = ("PASS", "Kiểm tra OK")
    ABNORMAL = ("ERROR_001", "Khay đĩa bất thường")
    ERR_NUM_DISK = ("ERROR_002", "Số lượng phát hiện khe đĩa bất thường")
