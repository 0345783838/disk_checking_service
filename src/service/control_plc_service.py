import time
import cv2
import numpy as np
from src.dtos.meta import DataResponse, ErrorCode, DataDebugResponse
from src.service.base_service import BaseService
import base64


class PlcControllingService(BaseService):
    def __init__(self):
        super().__init__()
        pass

    def connect_plc(self, ip, port):
        return self.plc_controller.connect(ip, port)

    def disconnect_plc(self):
        return self.plc_controller.disconnect()

    def turn_on_uv(self):
        return self.plc_controller.on_UV()

    def turn_off_uv(self):
        return self.plc_controller.off_UV()

    def turn_on_led_1(self):
        return self.plc_controller.on_LED_1()

    def turn_off_led_1(self):
        return self.plc_controller.off_LED_1()

    def turn_on_led_2(self):
        return self.plc_controller.on_LED_2()

    def turn_off_led_2(self):
        return self.plc_controller.off_LED_2()

    def check_connection(self):
        return self.plc_controller.check_connection()

    def read_trigger(self):
        return self.plc_controller.read_trigger()

    def reset_trigger(self):
        return self.plc_controller.reset_trigger()
