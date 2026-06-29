import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.lateral import apply_std_steer_angle_limits
from opendbc.car.subaru.carcontroller import (CarController, LKAS_ANGLE_LOW_SPEED_MAX,
                                              LKAS_ANGLE_YIELD_RELEASE, LKAS_ANGLE_LOW_SPEED)
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


class TestSubaruLowSpeedAngleClamp:
  """Drive the real controller and decode ES_LKAS_ANGLE to verify low-speed EPS fault avoidance: the
  active request is clamped just under the ~200 deg fault angle (request held, no chatter), and only
  once the measured wheel reaches the limit is the request dropped to track the measured angle."""

  LOW_SPEED = LKAS_ANGLE_LOW_SPEED - 0.5
  HIGH_SPEED = LKAS_ANGLE_LOW_SPEED + 0.5

  def setup_method(self):
    self.CP = CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT_2023)
    self.CP_SP = structs.CarParamsSP()
    self._reset_controller()

  def _reset_controller(self):
    self.cc = CarController(DBC[self.CP.carFingerprint], self.CP, self.CP_SP)
    # decode the angle command the controller transmits
    self.parser = CANParser(DBC[self.CP.carFingerprint][Bus.pt], [("ES_LKAS_ANGLE", 0)], CanBus.main)

  def _update(self, desired=150., measured=150., lat_active=True, v_ego=2.0):
    """Run one steering frame and return the decoded (LKAS_Output, LKAS_Request)."""
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
    # Steer-only frame (frame % STEER_STEP == 0, frame % 10 != 0): only ES_LKAS_ANGLE is emitted.
    self.cc.frame = 2
    _, can_sends = self.cc.update(CC.as_reader(), self.CP_SP, _CS(), 0)

    self.parser.update([0, can_sends])
    vl = self.parser.vl["ES_LKAS_ANGLE"]
    return vl["LKAS_Output"], vl["LKAS_Request"]

  def test_active_request_clamped_below_fault_angle(self):
    # below the speed threshold a large desired is clamped under the fault angle, request held
    for sign in (-1, 1):
      self._reset_controller()
      self.cc.apply_steer_last = sign * LKAS_ANGLE_LOW_SPEED_MAX
      output, request = self._update(desired=sign * 300., measured=sign * 150., v_ego=self.LOW_SPEED)
      assert request == 1
      assert not self.cc.lkas_angle_yield
      assert output == pytest.approx(sign * LKAS_ANGLE_LOW_SPEED_MAX, abs=0.05)

  def test_no_request_chatter_through_boundary(self):
    # the request stays set the whole way up to the limit (active command pinned at the clamp), then
    # drops exactly once when the measured wheel reaches it — no sawtooth/chatter
    self.cc.apply_steer_last = LKAS_ANGLE_LOW_SPEED_MAX
    measured = 150.
    toggles = 0
    prev_req = 1
    yielded_at = None
    for _ in range(60):
      output, request = self._update(desired=300., measured=measured, v_ego=self.LOW_SPEED)
      if request == 1:
        assert abs(output) <= LKAS_ANGLE_LOW_SPEED_MAX + 0.05   # active command never exceeds the limit
      if request != prev_req:
        toggles += 1
      if request == 0 and yielded_at is None:
        yielded_at = measured
      prev_req = request
      measured = min(measured + 2., 210.)  # wheel climbs toward and past the limit
    assert toggles == 1
    assert yielded_at >= LKAS_ANGLE_LOW_SPEED_MAX - 0.05

  def test_yield_when_measured_reaches_limit(self):
    for sign in (-1, 1):
      self._reset_controller()
      measured = sign * LKAS_ANGLE_LOW_SPEED_MAX
      self.cc.apply_steer_last = measured
      output, request = self._update(desired=sign * 300., measured=measured, v_ego=self.LOW_SPEED)
      assert request == 0
      assert output == pytest.approx(measured, abs=0.05)
      assert self.cc.lkas_angle_yield

  def test_large_measured_angle_yields_immediately(self):
    # Engaging near physical steering lock must not first send an active, mismatched angle.
    output, request = self._update(desired=0., measured=250., v_ego=self.LOW_SPEED)
    assert request == 0
    assert output == pytest.approx(250., abs=0.05)
    assert self.cc.lkas_angle_yield

  def test_yield_hysteresis(self):
    self.cc.apply_steer_last = LKAS_ANGLE_LOW_SPEED_MAX
    self._update(desired=300., measured=LKAS_ANGLE_LOW_SPEED_MAX, v_ego=self.LOW_SPEED)
    assert self.cc.lkas_angle_yield

    # Remaining above the release boundary holds the yield (no chatter).
    _, request = self._update(desired=0., measured=LKAS_ANGLE_YIELD_RELEASE + 5., v_ego=self.LOW_SPEED)
    assert request == 0
    assert self.cc.lkas_angle_yield

    # Dropping below the release boundary resumes active control.
    _, request = self._update(desired=0., measured=LKAS_ANGLE_YIELD_RELEASE - 10., v_ego=self.LOW_SPEED)
    assert request == 1
    assert not self.cc.lkas_angle_yield

  def test_high_speed_not_clamped_or_yielded(self):
    self.cc.apply_steer_last = 250.
    output, request = self._update(desired=250., measured=250., v_ego=self.HIGH_SPEED)
    assert request == 1
    assert not self.cc.lkas_angle_yield
    assert abs(output) > LKAS_ANGLE_LOW_SPEED_MAX  # not clamped above the speed threshold

  def test_inactive_resets_yield(self):
    self.cc.apply_steer_last = LKAS_ANGLE_LOW_SPEED_MAX
    self._update(desired=300., measured=LKAS_ANGLE_LOW_SPEED_MAX, v_ego=self.LOW_SPEED)
    assert self.cc.lkas_angle_yield

    _, request = self._update(desired=0., measured=200., lat_active=False)
    assert request == 0
    assert not self.cc.lkas_angle_yield

  def test_inactive_anchors_to_measured_angle(self):
    # when not active, send the measured angle without LKAS_Request, even at a large angle
    output, request = self._update(desired=0., measured=300., lat_active=False)
    assert request == 0
    assert output == pytest.approx(300., abs=0.05)


class TestSubaruParams:
  def test_ascent_steer_actuator_delays(self):
    assert CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT).steerActuatorDelay == pytest.approx(0.3)
    assert CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT_2023).steerActuatorDelay == pytest.approx(0.1)
