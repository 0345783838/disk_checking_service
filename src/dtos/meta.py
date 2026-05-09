from pydantic import BaseModel


class DataResponse(BaseModel):
    Result: int = 0  # 0/1/2 - ok/ng/warning
    ErrorCode: str = None  # ["PASS", "ERROR_001", "ERROR_002", "ERROR_003",...]
    ErrorDesc: str = None  # ["Khay đĩa OK", "Khay đĩa bất thường",...]
    ResImg: str = None  # base64 result encoded image
    MaxDiskDistance: float = None
    MinDiskDistance: float = None
    CropBox: str = None  # x1,x2,y1,y2
    UvBox1: str = None  # x1,x2,y1,y2
    UvBox2: str = None  # x1,x2,y1,y2
    Mid1: str = None
    Mid2: str = None


class DataResponseUv(BaseModel):
    Result: bool = False  # True/False
    ErrorCode: str = None  # ["PASS", "ERROR_001", "ERROR_002", "ERROR_003",...]
    ErrorDesc: str = None  # ["Khay đĩa OK", "Khay đĩa bất thường",...]
    CountUvDisk: int = None
    ResImg: str = None  # base64 result encoded image


class DataDebugResponse(BaseModel):
    Result: int = 0  # 0/1/2 - ok/ng/warning
    DetectImg: str = None  # base64 result encoded image
    SegmentImg: str = None  # base64 result encoded image
    FinalImg: str = None  # base64 result encoded image
    CropBox: str = None  # x1,x2,y1,y2
    UvBox1: str = None  # x1,x2,y1,y2
    UvBox2: str = None  # x1,x2,y1,y2
    Mid1: str = None
    Mid2: str = None


class DataDebugUVResponse(BaseModel):
    Result: bool = False  # True/False
    CountUvDisk: int = None
    ThresholdImg: str = None  # base64 result encoded image
    FinalImg: str = None  # base64 result encoded image


class ErrorCode:
    PASS = ("PASS", "Khay đĩa đạt chất lượng")
    ABNORMAL = ("ERROR_001", "Khay đĩa có bất thường!")
    WARNING_NUM_DISK = ("WARNING_001", "Số lượng đĩa trong khay đang thiếu!")
    ERR_NUM_DISK = ("ERROR_002", "Số lượng khe đĩa trong khay bất thường!")
    ERR_MIXING_DISK = ("ERROR_003", "Phát sinh mixing đĩa!")


class ClassifyResult:
    OK = 'ok'
    NG = 'ng'
    NO_DISK = 'no_disk'


class InspectionState:
    OK = 0
    NG = 1
    WARNING = 2
