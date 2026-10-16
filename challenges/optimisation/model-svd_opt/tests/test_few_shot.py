"""Shot-count and stopping rules exercised without acquiring simulator shots."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_workflow import Config
from model_svd_opt.few_shot import AdaptiveSession
from model_svd_opt.workflow import BudgetExhausted

class FakeMachine:
    def __init__(self,ratio=2.):
        self.calls=0
        self.last=({},dict(on_screen=True,sigma_x=ratio,sigma_y=ratio,on_screen_fraction=1.,transmission=1.,mu_x=0.,mu_y=0.,evaluations=0))
    def shot(self,settings,stage):
        self.calls+=1;info=dict(self.last[1],evaluations=self.calls);self.last=({},info);return self.last

class FakeSpace:
    start=np.array([.5])
    def settings(self,u):return dict(k=float(u[0]))

class AdaptiveStoppingTest(unittest.TestCase):
    def session(self,ratio=2.,budget=12):
        machine=FakeMachine(ratio);s=AdaptiveSession(machine,FakeSpace(),'optics',np.ones(2),Config(),budget);return machine,s
    def test_adequate_existing_reading_costs_zero_shots(self):
        machine,s=self.session(1.05);s.search('rcds',0);self.assertEqual(machine.calls,0);self.assertEqual(s.reason,'existing_reading_adequate')
    def test_seed_reading_is_reused_once(self):
        machine,s=self.session();s.measure([.5]);self.assertEqual(machine.calls,0);s.measure([.5]);self.assertEqual(machine.calls,1)
    def test_mbrl_patience_stops_after_three_nonimprovements(self):
        machine,s=self.session();s.measure([.5])
        for _ in range(3):s.measure([.6])
        self.assertEqual(machine.calls,3);self.assertEqual(s.reason,'no_significant_improvement')
        with self.assertRaises(BudgetExhausted):s.measure([.7])
        self.assertEqual(machine.calls,3)
    def test_actual_shot_cap_excludes_reused_seed(self):
        machine,s=self.session(budget=2);s.measure([.5]);s.measure([.6]);s.measure([.7])
        with self.assertRaises(BudgetExhausted):s.measure([.8])
        self.assertEqual(machine.calls,2)
    def test_mfbo_stops_when_expected_gain_is_below_cost(self):
        machine,s=self.session()
        class GP:
            def predict(self,x,return_std=True):return np.full(len(x),10.),np.full(len(x),.001)
        optimizer=SimpleNamespace(models=[GP()],space=SimpleNamespace(transform=lambda x:x))
        self.assertTrue(s.should_stop_bo(optimizer,[.5]));self.assertEqual(s.reason,'expected_improvement_below_shot_cost');self.assertEqual(machine.calls,0)

if __name__=='__main__':unittest.main()
