import numpy as np
import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.lateral import apply_std_steer_angle_limits
from opendbc.car.subaru.carcontroller import CarController, MAX_ANGLE_TRACKING_ERROR
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


class TestSubaruAngleTrackingClamp:
  """Drive CarController.update() for an LKAS_ANGLE car and decode the ES_LKAS_ANGLE frame to
  verify the tracking-error clamp: clamp the target to measured ±45° before rate-limiting, and
  drop LKAS_Request (sending the measured angle) when rate and tracking constraints are mutually
  impossible (e.g. rapid driver countersteer)."""

  LIMITS = CarControllerParams.ANGLE_LIMITS

  def setup_method(self):
    self.CP = CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT_2023)
    self.CP_SP = structs.CarParamsSP()
    self.cc = CarController(DBC[self.CP.carFingerprint], self.CP, self.CP_SP)
    # decode the angle command the controller transmits
    self.parser = CANParser(DBC[self.CP.carFingerprint][Bus.pt], [("ES_LKAS_ANGLE", 0)], CanBus.main)

  def _rate_up(self, v_ego: float) -> float:
    return float(np.interp(v_ego, self.LIMITS.ANGLE_RATE_LIMIT_UP[0], self.LIMITS.ANGLE_RATE_LIMIT_UP[1]))

  def _update(self, desired: float, measured: float, lat_active: bool = True, v_ego: float = 0., prev: float | None = None):
    """Run one steering frame and return the decoded (LKAS_Output, LKAS_Request)."""
    if prev is not None:
      self.cc.apply_steer_last = prev

    CC = structs.CarControl()
    CC.enabled = lat_active
    CC.latActive = lat_active
    CC.longActive = False
    CC.actuators.steeringAngleDeg = desired

    CS = structs.CarState()
    CS.steeringAngleDeg = measured
    CS.vEgoRaw = v_ego

    class _CS:
      out = CS
    # use a steer-only frame (frame % STEER_STEP == 0, frame % 10 != 0) so no dashboard
    # messages are emitted and only the ES_LKAS_ANGLE command is on the bus
    self.cc.frame = 2
    _, can_sends = self.cc.update(CC.as_reader(), self.CP_SP, _CS(), 0)

    self.parser.update([0, can_sends])
    vl = self.parser.vl["ES_LKAS_ANGLE"]
    return vl["LKAS_Output"], vl["LKAS_Request"]

  def test_clamp_before_rate_limit_respects_panda_budget(self):
    # large desired with measured reachable within the window: clamp first, then rate-limit, so the
    # active command steps by at most the per-frame rate (5° at 0 m/s) and stays within ±45°
    output, request = self._update(desired=300., measured=120., prev=100.)
    assert request == 1
    assert abs(output - 100.) <= self._rate_up(0.) + 0.05
    assert abs(output - 120.) <= MAX_ANGLE_TRACKING_ERROR + 0.05

  def test_clamp_before_rate_limit_negative(self):
    output, request = self._update(desired=-300., measured=-120., prev=-100.)
    assert request == 1
    assert abs(output - (-100.)) <= self._rate_up(0.) + 0.05
    assert abs(output - (-120.)) <= MAX_ANGLE_TRACKING_ERROR + 0.05

  def test_fallback_drops_request_on_fast_countersteer_positive(self):
    # rate limit from 100° only reaches 105°, tracking window minimum is 151°−45°=106° → mutually
    # impossible: drop LKAS_Request and send the measured angle to keep the EPS heartbeat alive
    output, request = self._update(desired=300., measured=151., prev=100.)
    assert request == 0
    assert output == pytest.approx(151., abs=0.05)

  def test_fallback_drops_request_on_fast_countersteer_negative(self):
    output, request = self._update(desired=-300., measured=-151., prev=-100.)
    assert request == 0
    assert output == pytest.approx(-151., abs=0.05)

  def test_no_fallback_during_normal_tight_turn(self):
    # tracking lag within the window: command is sent with LKAS_Request and stays within ±45°
    output, request = self._update(desired=200., measured=90., prev=100.)
    assert request == 1
    assert abs(output - 90.) <= MAX_ANGLE_TRACKING_ERROR + 0.05

  def test_inactive_heartbeat_extreme_angle_positive(self):
    # when not active, send the measured angle without LKAS_Request, even at ±482°
    output, request = self._update(desired=0., measured=482., lat_active=False)
    assert request == 0
    assert output == pytest.approx(482., abs=0.05)

  def test_inactive_heartbeat_extreme_angle_negative(self):
    output, request = self._update(desired=0., measured=-482., lat_active=False)
    assert request == 0
    assert output == pytest.approx(-482., abs=0.05)

  def test_consecutive_updates_never_exceed_rate(self):
    # several frames of a tight turn; the commanded angle never jumps more than the rate limit
    # and never leads the measured wheel angle by more than 45°
    prev = 0.
    measured = 0.
    rate_up = self._rate_up(0.)
    for _ in range(30):
      output, request = self._update(desired=300., measured=measured, prev=prev)
      assert abs(output - prev) <= rate_up + 0.05
      if request:
        assert abs(output - measured) <= MAX_ANGLE_TRACKING_ERROR + 0.05
      prev = output
      measured += 5.  # EPS tracking at ~5°/frame


class TestSubaruParams:
  def test_ascent_steer_actuator_delays(self):
    assert CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT).steerActuatorDelay == pytest.approx(0.3)
    assert CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT_2023).steerActuatorDelay == pytest.approx(0.1)
