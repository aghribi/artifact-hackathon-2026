import numpy as np
import requests
import time
from CATAP.magnet import MagnetFactory, Magnet
from CATAP.diagnostics.screen import ScreenFactory, Screen
from CATAP.diagnostics.camera import CameraFactory, Camera
 
from rl_framework.config import RLConfig
from rl_framework.envs.base_env import BaseAcceleratorEnv
 
import os
 
def _convert_value_to_current(magnet_name: str, value: float, momentum: float = 250.0) -> float:
    """
    Convert a native magnet value (k1 for quads, angle for correctors) to
    a current via the REST API at dsastvx10.dl.ac.uk:8001.
    """
    resp = requests.post(
        url="http://dsastvx10.dl.ac.uk:8001/magnet/convert",
        json={"current": value, "magnet_name": magnet_name, "momentum": momentum},
    )
    resp.raise_for_status()
    data = resp.json()
    return float(data["K"])
 
 
class RealEnv(BaseAcceleratorEnv):
    """
    Real-machine environment using CATAP for hardware magnets and screens.
    """
 
    def __init__(self, config: RLConfig, virtual: bool = False):
        super().__init__(config)
        if virtual:
            os.environ["EPICS_CA_ADDR_LIST"] = "192.168.83.246"
            os.environ["EPICS_CA_SERVER_PORT"] = "6040"
            os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
            os.environ["EPICS_CA_MAX_ARRAY_BYTES"] = "1000000"
        self._virtual = virtual
        self._magnet_factory = MagnetFactory(is_virtual=virtual)
        self._screen_factory = ScreenFactory(is_virtual=virtual)
        self._cam_factory = CameraFactory(is_virtual=virtual)
        print(virtual)
        # Build lookup: magnet name -> CATAP Magnet object
        self._catap_magnets: dict[str, Magnet] = {}
        for name in self._magnet_names:
            "cla_s04_mag_hvcor_01_h"
            claraname = name.replace("_","-").upper()
            if claraname[-2:] == "-H":
                claraname = claraname.replace("HVCOR","HCOR")[:-2]
            elif claraname[-2:] == "-V":
                claraname = claraname.replace("HVCOR","VCOR")[:-2]
            self._catap_magnets[name] = self._magnet_factory.get_magnet(claraname)
 
        # Build lookup: screen name -> CATAP Screen object
        self._catap_screens: dict[str, Screen] = {}
        self._catap_cams: dict[str, Camera] = {}
        scr_name = config.target_screen
        claraname = scr_name.replace("_","-").upper()
        self._catap_screens[scr_name] = self._screen_factory.get_screen(claraname)
        self._catap_cams[scr_name] = self._cam_factory.get_camera(self._catap_screens[scr_name].camera_name.replace("SCR","CAM"))
 
    def _hard_reset(self) -> np.ndarray:
        # Reset all magnets to zero current
        for name, magnet in self._catap_magnets.items():
            magnet.seti = (0.0)
        return self._get_obs()
 
    def _apply_action(self, native_values: np.ndarray):
        #TODO: handle splitting combined corrector values into horizontal and vertical components
        for i, name in enumerate(self._magnet_names):
            native_val = float(native_values[i])
            magnet = self._catap_magnets[name]
            current = _convert_value_to_current(name, native_val)
            magnet.seti = (current)
 
    def _get_screen_statistics(self, screen_name: str) -> tuple[float, float, float, float, bool]:
        screen:Screen = self._catap_screens[screen_name]
        if(not self._virtual and not screen.is_screen_in):
            screen.insert_yag()
            time.sleep(5)
            while screen.moving:
                print("waiting for screen to stop moving...")
                time.sleep(1) 
        cam:Camera = self._catap_cams[screen_name]
        if(not cam.is_acquiring):
            cam.start_acquiring()
        reading = cam.get_array_data()  # expected shape matches cheetah convention
        # TODO: look at getting stats with camera analysis? there's a bunch of catap functions for that
        return self._read_screen_stats(reading)