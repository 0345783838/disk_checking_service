import time

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.service.check_disk_service_yolo import DiskCheckingService
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


class UvParams(BaseModel):
    uv_disk_threshold: int
    uv_disk_min_area: float


class ParamsUv(BaseModel):
    threshold: float


class UvBox(BaseModel):
    crop_box: str
    uv_box_1: str
    uv_box_2: str
    mid_1: str
    mid_2: str


@inspection_router.post(path='/check_disk_white')
def check_disk(image: UploadFile = File(...)):
    time_st = time.time()
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
    res = disk_checking_service.check_disk_white(img)
    print("Time white light ms = ", (time.time() - time_st)*1000)
    return res


@inspection_router.post(path='/check_disk_uv')
def check_disk(image: UploadFile = File(...), uv_box: str = Form(...)):
    time_st = time.time()
    if not image.file or not uv_box:
        raise HTTPException(status_code=400, detail="Invalid input")
    uv_box_json = UvBox(**json.loads(uv_box))
    crop_box = uv_box_json.crop_box
    uv_box_1 = uv_box_json.uv_box_1
    uv_box_2 = uv_box_json.uv_box_2
    mid_1 = uv_box_json.mid_1
    mid_2 = uv_box_json.mid_2

    if not uv_box_1 or not uv_box_2 or not mid_1 or not mid_1 or not mid_2:
        raise HTTPException(status_code=400, detail="Invalid input")

    img_str = image.file.read()
    if img_str is None or img_str == b'':
        # Cannot read image
        raise HTTPException(status_code=400, detail="Invalid input")
    try:
        np_img = np.fromstring(img_str, np.uint8)
        img = cv2.imdecode(np_img, flags=1)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid input")
    except Exception as ex:
        # Cannot decode image
        raise HTTPException(status_code=400, detail="Invalid input")
    res = disk_checking_service.check_disk_uv(img, crop_box, uv_box_1, uv_box_2, mid_1, mid_2)
    print("Time uv light ms = ", (time.time() - time_st)*1000)
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


@inspection_router.post(path='/check_disk_uv_debug')
def check_disk_uv_debug(image: UploadFile = File(...), params_json: str = Form(...)):
    if not image.file:
        raise HTTPException(status_code=400, detail="Invalid input")
    if not params_json:
        raise HTTPException(status_code=400, detail="Invalid input")

    params = UvParams(**json.loads(params_json))
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

    res = disk_checking_service.check_disk_uv_debug(img, params)
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


@communication_router.get(path='/control_uv_1')
def control_uv(status: bool = Query(...)):
    return {"Success": plc_controlling_service.turn_on_uv_1() if status else plc_controlling_service.turn_off_uv_1()}


@communication_router.get(path='/control_uv_2')
def control_uv(status: bool = Query(...)):
    return {"Success": plc_controlling_service.turn_on_uv_2() if status else plc_controlling_service.turn_off_uv_2()}


@communication_router.get(path='/control_led_1')
def control_led_1(status: bool = Query(...)):
    return {"Success": plc_controlling_service.turn_on_led_1() if status else plc_controlling_service.turn_off_led_1()}


@communication_router.get(path='/control_led_2')
def control_led_2(status: bool = Query(...)):
    return {"Success": plc_controlling_service.turn_on_led_2() if status else plc_controlling_service.turn_off_led_2()}


@communication_router.get(path='/check_connection')
def check_connection():
    return {"Success": plc_controlling_service.check_connection()}


@communication_router.get(path='/read_trigger')
def read_trigger():
    return {"Success": plc_controlling_service.read_trigger()[0],
            "Status": plc_controlling_service.read_trigger()[1]}


@communication_router.get(path='/reset_trigger')
def reset_trigger():
    return {"Success": plc_controlling_service.reset_trigger()}


@communication_router.get(path='/on_error')
def on_error():
    return {"Success": plc_controlling_service.plc_controller.on_error()}
