from pydantic import BaseModel


class DataResponse(BaseModel):
    Result: bool = False  # True/False
    ErrorCode: str = None  # ["PASS", "ERROR_001", "ERROR_002", "ERROR_003",...]
    ErrorDesc: str = None  # ["Khay đĩa OK", "Khay đĩa bất thường",...]
    ResImg: str = None  # base64 result encoded image
    MaxDiskDistance: float = None
    MinDiskDistance: float = None
    CropBox: str = None  # x1,x2,y1,y2
    UvBox1: str = None  # x1,x2,y1,y2
    UvBox2: str = None  # x1,x2,y1,y2


class DataResponseUv(BaseModel):
    Result: bool = False  # True/False
    ErrorCode: str = None  # ["PASS", "ERROR_001", "ERROR_002", "ERROR_003",...]
    ErrorDesc: str = None  # ["Khay đĩa OK", "Khay đĩa bất thường",...]
    CountUvDisk: int = None
    ResImg: str = None  # base64 result encoded image


class DataDebugResponse(BaseModel):
    Result: bool = False  # True/False
    DetectImg: str = None  # base64 result encoded image
    SegmentImg: str = None  # base64 result encoded image
    FinalImg: str = None  # base64 result encoded image


class ErrorCode:
    PASS = ("PASS", "Khay đĩa đạt chất lượng")
    ABNORMAL = ("ERROR_001", "Khay đĩa có bất thường")
    ERR_NUM_DISK = ("ERROR_002", "Số lượng khe đĩa trong khay bất thường")
    ERR_NUM_UV_DISK = ("ERROR_003", "Có đĩa UV trong khay")
