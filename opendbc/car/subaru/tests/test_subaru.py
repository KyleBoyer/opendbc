import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.lateral import apply_std_steer_angle_limits
from opendbc.car.subaru.carcontroller import (CarController, EPS_TORQUE_HIGH, EPS_TORQUE_RELEASE,
                                              EPS_TORQUE_FAULT_FRAMES, EPS_INHIBIT_MIN_FRAMES, EPS_RELEASE_FRAMES)
from opendbc.car.subaru.fingerprints import FW_VERSIONS
from opendbc.car.subaru.interface import CarInterface
from opendbc.car.subaru.values import CAR, CanBus, CarControllerParams, DBC


class TestSubaruFingerprint:
  def test_fw_version_format(self):
    for platform, fws_per_ecu in FW_VERSIONS.items():
      for (ecu, _, _), fws in fws_per_ecu.items():
        fw_size = len(fws[0])
        for fw in fws:
          assert len(fw) == fw_size, f"{platform} {ecu}: {len(fw)} {fw_size}"


class TestSubaruAngleLimits:
  def test_full_low_speed_angle_range(self):
    limits = CarControllerParams.ANGLE_LIMITS

    assert apply_std_steer_angle_limits(600, 540, 0, 0, True, limits) == 545
    assert apply_std_steer_angle_limits(-600, -540, 0, 0, True, limits) == -545


class TestSubaruEpsTorqueInhibitor:
  """Drive CarController.update() for an LKAS_ANGLE car and decode the ES_LKAS_ANGLE frame to verify
  the EPS effort inhibitor: when the EPS self-drive torque (steeringTorqueEps) is sustained near its
  fault ceiling, the controller drops LKAS_Request and anchors the command to the measured angle,
  holding the cutout with hysteresis until torque relaxes."""

  def setup_method(self):
    self.CP = CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT_2023)
    self.CP_SP = structs.CarParamsSP()
    self.cc = CarController(DBC[self.CP.carFingerprint], self.CP, self.CP_SP)
    # decode the angle command the controller transmits
    self.parser = CANParser(DBC[self.CP.carFingerprint][Bus.pt], [("ES_LKAS_ANGLE", 0)], CanBus.main)

  def _update(self, desired=150., measured=150., eps_torque=0., lat_active=True, v_ego=2.0):
    """Run one steering frame and return the decoded (LKAS_Output, LKAS_Request)."""
    CC = structs.CarControl()
    CC.enabled = lat_active
    CC.latActive = lat_active
    CC.longActive = False
    CC.actuators.steeringAngleDeg = desired

    CS = structs.CarState()
    CS.steeringAngleDeg = measured
    CS.vEgoRaw = v_ego
    CS.steeringTorqueEps = eps_torque

    class _CS:
      out = CS
    # steer-only frame (frame % STEER_STEP == 0, frame % 10 != 0): only ES_LKAS_ANGLE is emitted,
    # and the inhibitor counters advance one steering update per call
    self.cc.frame = 2
    _, can_sends = self.cc.update(CC.as_reader(), self.CP_SP, _CS(), 0)

    self.parser.update([0, can_sends])
    vl = self.parser.vl["ES_LKAS_ANGLE"]
    return vl["LKAS_Output"], vl["LKAS_Request"]

  def _run(self, n, **kw):
    out = req = None
    for _ in range(n):
      out, req = self._update(**kw)
    return out, req

  def test_normal_torque_passes_through(self):
    # ordinary active driving keeps the request and tracks the commanded angle
    _, request = self._update(desired=10., measured=10., eps_torque=500.)
    assert request == 1

  def test_low_torque_never_inhibits(self):
    # sustained ordinary torque (well below the ceiling) never trips the inhibitor
    _, request = self._run(EPS_TORQUE_FAULT_FRAMES * 3, desired=10., measured=10., eps_torque=600.)
    assert request == 1
    assert not self.cc.eps_inhibit

  def test_brief_high_torque_does_not_inhibit(self):
    # a high-torque spike shorter than the trigger window is ignored
    self._run(EPS_TORQUE_FAULT_FRAMES - 1, desired=200., measured=150., eps_torque=EPS_TORQUE_HIGH + 300)
    _, request = self._update(desired=200., measured=150., eps_torque=400.)
    assert request == 1
    assert not self.cc.eps_inhibit

  def test_sustained_high_torque_inhibits(self):
    high = EPS_TORQUE_HIGH + 300
    # one frame short of the trigger: still active
    _, request = self._run(EPS_TORQUE_FAULT_FRAMES - 1, desired=200., measured=150., eps_torque=high)
    assert request == 1
    # the trigger frame drops the request and anchors the command to the measured angle
    output, request = self._update(desired=200., measured=150., eps_torque=high)
    assert request == 0
    assert output == pytest.approx(150., abs=0.05)
    assert self.cc.eps_inhibit

  def test_inhibit_holds_while_torque_stays_elevated(self):
    # trip, then feed torque between the release and high thresholds: never releases
    self._run(EPS_TORQUE_FAULT_FRAMES, desired=200., measured=150., eps_torque=EPS_TORQUE_HIGH + 300)
    assert self.cc.eps_inhibit
    mid = (EPS_TORQUE_HIGH + EPS_TORQUE_RELEASE) // 2
    _, request = self._run(EPS_INHIBIT_MIN_FRAMES + EPS_RELEASE_FRAMES + 5, desired=200., measured=150., eps_torque=mid)
    assert request == 0
    assert self.cc.eps_inhibit

  def test_release_requires_min_hold_and_low_torque(self):
    self._run(EPS_TORQUE_FAULT_FRAMES, desired=200., measured=150., eps_torque=EPS_TORQUE_HIGH + 300)
    assert self.cc.eps_inhibit
    low = EPS_TORQUE_RELEASE - 200
    # low torque but short of the minimum hold: still inhibited
    _, request = self._run(EPS_INHIBIT_MIN_FRAMES - 1, desired=200., measured=150., eps_torque=low)
    assert request == 0
    assert self.cc.eps_inhibit
    # the frame that satisfies both the min hold and the low-torque dwell re-engages
    _, request = self._update(desired=200., measured=150., eps_torque=low)
    assert request == 1
    assert not self.cc.eps_inhibit

  def test_inactive_resets_inhibition(self):
    self._run(EPS_TORQUE_FAULT_FRAMES, desired=200., measured=150., eps_torque=EPS_TORQUE_HIGH + 300)
    assert self.cc.eps_inhibit
    # going inactive (driver override / disengage) clears the cutout immediately
    _, request = self._update(desired=0., measured=150., eps_torque=2000., lat_active=False)
    assert request == 0
    assert not self.cc.eps_inhibit

  def test_inactive_anchors_to_measured_angle(self):
    # when not active, send the measured angle without LKAS_Request, even at a large angle
    output, request = self._update(desired=0., measured=196., eps_torque=1700., lat_active=False)
    assert request == 0
    assert output == pytest.approx(196., abs=0.05)


class TestSubaruParams:
  def test_ascent_steer_actuator_delays(self):
    assert CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT).steerActuatorDelay == pytest.approx(0.3)
    assert CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT_2023).steerActuatorDelay == pytest.approx(0.1)
