"""Calibration freshness checks; interface compatibility does not imply physics validity."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
from typing import Protocol,Mapping,Any


class OpticsProvider(Protocol):
    def evaluate(self,sigma_x_m:float,sigma_y_m:float,**kwargs)->Mapping[str,Any]: ...
    def configuration_fingerprint(self)->str: ...


class StaleCalibrationError(RuntimeError):
    """The public API is usable, but the saved physical response needs recalibration."""


@lru_cache(maxsize=32)
def _digest(path,mtime,size):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_digest(path):
    p=Path(path);s=p.stat()
    return _digest(str(p.resolve()),s.st_mtime_ns,s.st_size)


def optics_fingerprint(optics=None):
    if optics is None:
        from optics_matching_vbc_on import configuration_fingerprint
        return configuration_fingerprint(l4h_off=True)
    method=getattr(optics,'configuration_fingerprint',None)
    if not callable(method):raise TypeError('An injected optics provider must implement configuration_fingerprint() and evaluate(...)["settings"]')
    value=method()
    if not isinstance(value,str) or not value:raise TypeError('configuration_fingerprint() must return a non-empty string')
    return value


def record_calibration(directory,optics=None):
    directory=Path(directory)
    report=dict(schema_version=1,optics_fingerprint=optics_fingerprint(optics),
        response_sha256=file_digest(directory/'orbit_response_model.npz'))
    (directory/'calibration.json').write_text(json.dumps(report,indent=2)+'\n')


def record_modes(directory):
    directory=Path(directory)
    report=dict(schema_version=1,response_sha256=file_digest(directory/'orbit_response_model.npz'),
        mode_sha256={p:file_digest(directory/f'orbit_modes_{p}.npz') for p in 'xy'})
    (directory/'mode_calibration.json').write_text(json.dumps(report,indent=2)+'\n')


class CalibrationGuard:
    def __init__(self,directory,optics=None):
        self.directory=Path(directory);self.optics=optics
    def check(self):
        try:expected=json.loads((self.directory/'calibration.json').read_text())
        except (FileNotFoundError,ValueError) as exc:raise StaleCalibrationError('Missing calibration identity. Recalibrate with python -m orbit_matching.build_vbc_orbits.') from exc
        if expected['optics_fingerprint']!=optics_fingerprint(self.optics) or expected['response_sha256']!=file_digest(self.directory/'orbit_response_model.npz'):
            raise StaleCalibrationError('Optics configuration or response changed. Re-evaluate with python -m orbit_matching.build_vbc_orbits.')


def check_modes(path):
    path=Path(path);directory=path.parent
    try:expected=json.loads((directory/'mode_calibration.json').read_text())
    except (FileNotFoundError,ValueError) as exc:raise StaleCalibrationError('Missing mode calibration identity. Run python -m orbit_matching.build_vbc_orbits.') from exc
    plane=path.stem.rsplit('_',1)[-1]
    if expected['response_sha256']!=file_digest(directory/'orbit_response_model.npz') or expected['mode_sha256'].get(plane)!=file_digest(path):
        raise StaleCalibrationError('Saved SVD bundle is stale. Run python -m orbit_matching.build_vbc_orbits after response recalibration.')
