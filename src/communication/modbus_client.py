from pymodbus.client import ModbusTcpClient
import traceback
import time


class MbClient:
    # ip="192.168.0.211", port=8000,
    def __init__(self, id=1):
        self.HMI_IP = None
        self.PORT = None
        self.UNIT_ID = id
        self.client = None

    def connect(self, ip, port):
        try:
            self.client = ModbusTcpClient(ip, port=port, timeout=1)
            if self.client.connect():
                self.HMI_IP = ip
                self.PORT = port
                return True
            else:
                return False

        except Exception:
            print("[Mb Client] Exception", traceback.format_exc())
            return False

    def disconnect(self):
        if self.client:
            try:
                self.client.close()
                return True
            except Exception:
                return False
        return True

    # ================= CHECK CONNECTION =================
    def check_connection(self) -> bool:
        """
        Kiểm tra kết nối Modbus còn sống hay không
        """
        if not self.client:
            return False

        try:
            rr = self.client.read_coils(0, 1, unit=self.UNIT_ID)
            if rr is None or rr.isError():
                return False

            return True

        except Exception:
            return False

    # ====================================================

    def on_UV(self):
        return self.__write_bit(2, True)

    def off_UV(self):
        return self.__write_bit(2, False)

    def on_LED(self):
        return self.__write_bit(3, True)

    def off_LED(self):
        return self.__write_bit(3, False)

    def on_error(self):
        return self.__write_bit(1, True)

    def off_error(self):
        return self.__write_bit(1, False)

    def __write_bit(self, addr: int, value: bool = False):
        try:
            if not self.check_connection():
                return False

            rr = self.client.write_coil(addr, value, unit=self.UNIT_ID)
            if rr.isError():
                return False

            return True

        except Exception:
            return False


if __name__ == "__main__":
    pass
