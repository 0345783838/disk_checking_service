from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.service.check_disk_service import DiskCheckingService
from typing import List
import numpy as np
import json
import cv2
import io

from src.service.control_plc_service import PlcControllingService

# diamond_router = APIRouter(dependencies=[Depends(check_auth)])
inspection_router = APIRouter()
communication_router = APIRouter()
disk_checking_service = DiskCheckingService()
plc_controlling_service = PlcControllingService()


class Params(BaseModel):
    segment_threshold: float

    detect_threshold: float
    detect_iou: float

    caliper_min_edge_distance: float
    caliper_max_edge_distance: float
    caliper_length_rate: float
    caliper_thickness_list: List[int]

    disk_num: int
    disk_max_distance: float
    disk_min_distance: float
    disk_min_area: float


@inspection_router.post(path='/check_disk')
def check_disk(image: UploadFile = File(...)):
    if not image.file:
        raise HTTPException(status_code=400, detail="Invalid input")

    img_str = image.file.read()
    if img_str is None or img_str == b'':
        # Cannot read image
        return HTTPException(status_code=400, detail="Invalid input")
    try:
        np_img = np.fromstring(img_str, np.uint8)
        img = cv2.imdecode(np_img, flags=1)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid input")
    except Exception as ex:
        # Cannot decode image
        raise HTTPException(status_code=400, detail="Invalid input")
    res = disk_checking_service.check_disk(img)
    return res


@inspection_router.post(path='/check_disk_swagger')
def check_disk_swagger(image: UploadFile = File(...)):
    if not image.file:
        raise HTTPException(status_code=400, detail="Invalid input")

    img_str = image.file.read()
    if img_str is None or img_str == b'':
        # Cannot read image
        return HTTPException(status_code=400, detail="Invalid input")
    try:
        np_img = np.fromstring(img_str, np.uint8)
        img = cv2.imdecode(np_img, flags=1)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid input")
    except Exception as ex:
        # Cannot decode image
        raise HTTPException(status_code=400, detail="Invalid input")

    draw_image = disk_checking_service.check_disk_swagger(img)
    _, encoded_image = cv2.imencode('.jpg', draw_image)
    image_bytes = io.BytesIO(encoded_image.tobytes())

    return StreamingResponse(image_bytes, media_type="image/jpeg")


@inspection_router.post(path='/check_disk_debug')
def check_disk_debug(image: UploadFile = File(...), params_json: str = Form(...)):
    if not image.file:
        raise HTTPException(status_code=400, detail="Invalid input")
    if not params_json:
        raise HTTPException(status_code=400, detail="Invalid input")

    params = Params(**json.loads(params_json))
    img_str = image.file.read()
    if img_str is None or img_str == b'':
        # Cannot read image
        return HTTPException(status_code=400, detail="Invalid input")
    try:
        np_img = np.fromstring(img_str, np.uint8)
        img = cv2.imdecode(np_img, flags=1)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid input")
    except Exception as ex:
        # Cannot decode image
        raise HTTPException(status_code=400, detail="Invalid input")

    res = disk_checking_service.check_disk_debug(img, params)
    return res


@inspection_router.get(path='/check_service_status')
def check_service_status():
    return {"status": "running"}


@communication_router.get(path='/connect_plc')
def connect_plc(ip: str, port: int):
    return {"Success": plc_controlling_service.connect_plc(ip, port)}


@communication_router.get(path='/disconnect_plc')
def disconnect_plc():
    return {"Success": plc_controlling_service.disconnect_plc()}


@communication_router.get(path='/control_uv')
def control_uv(status: bool = Query(...)):
    return {"Success": plc_controlling_service.turn_on_uv() if status else plc_controlling_service.turn_off_uv()}


@communication_router.get(path='/control_led')
def control_led(status: bool= Query(...)):
    return {"Success": plc_controlling_service.turn_on_led() if status else plc_controlling_service.turn_off_led()}


@communication_router.get(path='/check_connection')
def check_connection():
    return {"Success": plc_controlling_service.check_connection()}


@communication_router.get(path='/read_trigger')
def read_trigger():
    return {"Success": plc_controlling_service.read_trigger()}


@communication_router.get(path='/on_error')
def on_error():
    return {"Success": plc_controlling_service.plc_controller.on_error()}
