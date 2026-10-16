"""Run with python -m unittest orbit_matching.test_compatibility."""
import unittest
from unittest.mock import patch
import numpy as np
from . import OrbitModes,StaleCalibrationError


class CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models={p:OrbitModes(p) for p in 'xy'}
        cls.x=float(cls.models['x'].axis[5]);cls.y=float(cls.models['x'].axis[7])

    def test_legacy_imports(self):
        from optics_matching import OrbitModes as legacy
        self.assertIs(legacy,OrbitModes)

    def test_reordered_optics_provider(self):
        from optics_matching_vbc_on import MatchedOptics
        base=MatchedOptics(l4h_off=True)
        class Reordered:
            def configuration_fingerprint(self):return base.configuration_fingerprint()
            def evaluate(self,x,y):
                tune=base.evaluate(x,y)
                return dict(settings=dict(reversed(list(tune['settings'].items()))))
        model=OrbitModes('x',optics=Reordered(),backend='cheetah')
        np.testing.assert_allclose(model.evaluate(.001,.003)['weighted_response'],OrbitModes('x',backend='cheetah').evaluate(.001,.003)['weighted_response'],rtol=0,atol=0)

    def test_stale_optics_blocks_existing_models_and_raw_splines(self):
        with patch('optics_matching_vbc_on.model.configuration_fingerprint',return_value='modified-optics'):
            for p in 'xy':
                with self.assertRaises(StaleCalibrationError):self.models[p].evaluate(.001,.003)
                with self.assertRaises(StaleCalibrationError):self.models[p].evaluate(.001,.003,exact_response=False)
                with self.assertRaises(StaleCalibrationError):OrbitModes(p)

    def test_changed_response_invalidates_saved_modes(self):
        from .compatibility import check_modes
        with patch('orbit_matching.compatibility.file_digest',return_value='new-response'):
            with self.assertRaises(StaleCalibrationError):check_modes(self.models['x']._bundle_path)

    def test_modal_roundtrip(self):
        for p,m in self.models.items():
            amplitude=np.array([1.,-2.,3.])*1e-6
            result=m.from_kicks(m.to_kicks(amplitude,self.x,self.y),self.x,self.y)
            np.testing.assert_allclose(result[:3],amplitude,rtol=0,atol=1e-18)

    def test_backend_adapters(self):
        import optics_matching_vbc_on as optics_matching
        from . import apply_to_xsuite,apply_to_cheetah
        x,y=self.x,self.y
        from optics_matching_vbc_on import MatchedOptics
        optics=MatchedOptics(l4h_off=True)
        converted=optics_matching.build_xsuite(x,y,model=optics)
        segment=optics_matching.build_cheetah(x,y,model=optics)
        elements={e.name:e for e in segment.elements}
        for p,m in self.models.items():
            field='knl' if p=='x' else 'ksl';angle='horizontal_angle' if p=='x' else 'vertical_angle';sign=-1 if p=='x' else 1
            original=np.array([float(getattr(converted.line[n],field)[0]) for n in m.correctors])
            old=np.array([float(getattr(elements[n],angle)) for n in m.correctors])
            expected=m.to_kicks([1e-6,-2e-6],x,y)
            apply_to_xsuite(converted,x,y,[1e-6,-2e-6],plane=p,model=m)
            apply_to_cheetah(segment,x,y,[1e-6,-2e-6],plane=p,model=m)
            np.testing.assert_allclose([getattr(converted.line[n],field)[0] for n in m.correctors],original+sign*expected,rtol=0,atol=1e-18)
            np.testing.assert_allclose([float(getattr(elements[n],angle)) for n in m.correctors],old+expected,rtol=0,atol=1e-12)

if __name__=='__main__':unittest.main()
