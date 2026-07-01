import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.lateral import apply_std_steer_angle_limits
from opendbc.car.subaru.carcontroller import (CarController, LKAS_ANGLE_MAX_ACTIVE, LKAS_ANGLE_YIELD_RELEASE,
                                              STEER_OVERRIDE_TORQUE_HIGH, STEER_OVERRIDE_TORQUE_LOW)
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


class TestSubaruAngleClamp:
  """Drive the real controller and decode ES_LKAS_ANGLE to verify EPS fault avoidance: the active
  request is clamped just under the ~200 deg fault angle (request held, no chatter) at all speeds,
  and only once the measured wheel reaches the limit is the request dropped to track measured."""

  LOW_SPEED = 2.0   # m/s
  HIGH_SPEED = 8.0  # m/s (well above the old 10 mph gate; the clamp is speed-independent)

  def setup_method(self):
    self.CP = CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT_2023)
    self.CP_SP = structs.CarParamsSP()
    self._reset_controller()

  def _reset_controller(self):
    self.cc = CarController(DBC[self.CP.carFingerprint], self.CP, self.CP_SP)
    # decode the angle command the controller transmits
    self.parser = CANParser(DBC[self.CP.carFingerprint][Bus.pt], [("ES_LKAS_ANGLE", 0)], CanBus.main)

  def _update(self, desired=150., measured=150., lat_active=True, v_ego=2.0, driver_torque=0., directional_override=True):
    """Run one steering frame and return the decoded (LKAS_Output, LKAS_Request)."""
    CC = structs.CarControl()
    CC.enabled = lat_active
    CC.latActive = lat_active
    CC.longActive = False
    CC.actuators.steeringAngleDeg = desired

    CC_SP = structs.CarControlSP()
    CC_SP.subaruDirectionalSteerOverride = directional_override

    CS = structs.CarState()
    CS.steeringAngleDeg = measured
    CS.vEgoRaw = v_ego
    CS.steeringTorque = driver_torque

    class _CS:
      out = CS
    # Steer-only frame (frame % STEER_STEP == 0, frame % 10 != 0): only ES_LKAS_ANGLE is emitted.
    self.cc.frame = 2
    _, can_sends = self.cc.update(CC.as_reader(), CC_SP, _CS(), 0)

    self.parser.update([0, can_sends])
    vl = self.parser.vl["ES_LKAS_ANGLE"]
    return vl["LKAS_Output"], vl["LKAS_Request"]

  def test_active_request_clamped_below_fault_angle(self):
    # a large desired is clamped under the fault angle with the request held
    for sign in (-1, 1):
      self._reset_controller()
      self.cc.apply_steer_last = sign * LKAS_ANGLE_MAX_ACTIVE
      output, request = self._update(desired=sign * 300., measured=sign * 150., v_ego=self.LOW_SPEED)
      assert request == 1
      assert not self.cc.lkas_angle_yield
      assert output == pytest.approx(sign * LKAS_ANGLE_MAX_ACTIVE, abs=0.05)

  def test_no_request_chatter_through_boundary(self):
    # the request stays set the whole way up to the limit (active command pinned at the clamp), then
    # drops exactly once when the measured wheel reaches it — no sawtooth/chatter
    self.cc.apply_steer_last = LKAS_ANGLE_MAX_ACTIVE
    measured = 150.
    toggles = 0
    prev_req = 1
    yielded_at = None
    for _ in range(60):
      output, request = self._update(desired=300., measured=measured, v_ego=self.LOW_SPEED)
      if request == 1:
        assert abs(output) <= LKAS_ANGLE_MAX_ACTIVE + 0.05   # active command never exceeds the limit
      if request != prev_req:
        toggles += 1
      if request == 0 and yielded_at is None:
        yielded_at = measured
      prev_req = request
      measured = min(measured + 2., 210.)  # wheel climbs toward and past the limit
    assert toggles == 1
    assert yielded_at >= LKAS_ANGLE_MAX_ACTIVE - 0.05

  def test_yield_when_measured_reaches_limit(self):
    for sign in (-1, 1):
      self._reset_controller()
      measured = sign * LKAS_ANGLE_MAX_ACTIVE
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
    self.cc.apply_steer_last = LKAS_ANGLE_MAX_ACTIVE
    self._update(desired=300., measured=LKAS_ANGLE_MAX_ACTIVE, v_ego=self.LOW_SPEED)
    assert self.cc.lkas_angle_yield

    # Remaining above the release boundary holds the yield (no chatter).
    _, request = self._update(desired=0., measured=LKAS_ANGLE_YIELD_RELEASE + 5., v_ego=self.LOW_SPEED)
    assert request == 0
    assert self.cc.lkas_angle_yield

    # Dropping below the release boundary resumes active control.
    _, request = self._update(desired=0., measured=LKAS_ANGLE_YIELD_RELEASE - 10., v_ego=self.LOW_SPEED)
    assert request == 1
    assert not self.cc.lkas_angle_yield

  def test_yield_holds_while_model_still_requesting(self):
    # while the model still requests a large angle, the yield holds even if the measured wheel dips
    # below the release boundary (overshoot/exit), so we don't re-engage and jerk the wheel
    self.cc.apply_steer_last = LKAS_ANGLE_MAX_ACTIVE
    self._update(desired=300., measured=LKAS_ANGLE_MAX_ACTIVE, v_ego=self.LOW_SPEED)
    assert self.cc.lkas_angle_yield

    _, request = self._update(desired=300., measured=LKAS_ANGLE_YIELD_RELEASE - 20., v_ego=self.LOW_SPEED)
    assert request == 0
    assert self.cc.lkas_angle_yield

    # only once the model request also eases (turn ending) and the wheel is down do we resume
    _, request = self._update(desired=LKAS_ANGLE_YIELD_RELEASE - 20., measured=LKAS_ANGLE_YIELD_RELEASE - 20.,
                              v_ego=self.LOW_SPEED)
    assert request == 1
    assert not self.cc.lkas_angle_yield

  def test_clamp_applies_at_high_speed(self):
    # the clamp is speed-independent: a large desired well above the old 10 mph gate is still clamped
    self.cc.apply_steer_last = LKAS_ANGLE_MAX_ACTIVE
    output, request = self._update(desired=300., measured=150., v_ego=self.HIGH_SPEED)
    assert request == 1
    assert output == pytest.approx(LKAS_ANGLE_MAX_ACTIVE, abs=0.05)

  def test_inactive_resets_yield(self):
    self.cc.apply_steer_last = LKAS_ANGLE_MAX_ACTIVE
    self._update(desired=300., measured=LKAS_ANGLE_MAX_ACTIVE, v_ego=self.LOW_SPEED)
    assert self.cc.lkas_angle_yield

    _, request = self._update(desired=0., measured=200., lat_active=False)
    assert request == 0
    assert not self.cc.lkas_angle_yield

  def test_inactive_anchors_to_measured_angle(self):
    # when not active, send the measured angle without LKAS_Request, even at a large angle
    output, request = self._update(desired=0., measured=300., lat_active=False)
    assert request == 0
    assert output == pytest.approx(300., abs=0.05)

  def test_driver_override_drops_request(self):
    # high opposing driver torque drops the active request (anchored to measured) so the driver
    # steers freely - desired is positive (turning right), torque is negative (driver pulling back)
    self.cc.apply_steer_last = 150.
    output, request = self._update(desired=250., measured=150., driver_torque=-(STEER_OVERRIDE_TORQUE_HIGH + 20.))
    assert request == 0
    assert self.cc.driver_override
    assert output == pytest.approx(150., abs=0.05)

    # hysteresis: torque between the thresholds keeps the override engaged
    _, request = self._update(desired=250., measured=150.,
                              driver_torque=-((STEER_OVERRIDE_TORQUE_HIGH + STEER_OVERRIDE_TORQUE_LOW) // 2))
    assert request == 0
    assert self.cc.driver_override

    # once torque is released, active control resumes
    _, request = self._update(desired=250., measured=150., driver_torque=-(STEER_OVERRIDE_TORQUE_LOW - 20.))
    assert request == 1
    assert not self.cc.driver_override

  def test_directional_override_ignores_same_direction_torque(self):
    # with directional override on (default), high torque in the SAME direction as the commanded
    # angle (helping the turn) must not drop the request
    _, request = self._update(desired=250., measured=150., driver_torque=STEER_OVERRIDE_TORQUE_HIGH + 50.,
                              directional_override=True)
    assert request == 1
    assert not self.cc.driver_override

  def test_directional_override_disabled_is_magnitude_only(self):
    # with the toggle off, same-direction high torque still drops the request (legacy behavior)
    output, request = self._update(desired=250., measured=150., driver_torque=STEER_OVERRIDE_TORQUE_HIGH + 50.,
                                   directional_override=False)
    assert request == 0
    assert self.cc.driver_override
    assert output == pytest.approx(150., abs=0.05)

  def test_directional_override_near_zero_angle_falls_back_to_magnitude(self):
    # commanded angle near straight-ahead: direction is unreliable, so any high torque overrides
    output, request = self._update(desired=2., measured=0., driver_torque=STEER_OVERRIDE_TORQUE_HIGH + 50.,
                                   directional_override=True)
    assert request == 0
    assert self.cc.driver_override

  def test_directional_override_unwind_assist_not_misclassified(self):
    # model is unwinding (desired=100 < measured=150, both positive - reducing the turn) and the
    # driver helps by pushing toward center (negative torque, matching the requested motion toward
    # desired) - even though desired_angle itself is still positive, comparing against the requested
    # motion (desired - measured) rather than raw desired_angle must not misclassify this as opposing
    _, request = self._update(desired=100., measured=150., driver_torque=-(STEER_OVERRIDE_TORQUE_HIGH + 50.),
                              directional_override=True)
    assert request == 1
    assert not self.cc.driver_override

  def test_directional_override_clears_on_direction_flip_without_dropping_below_low(self):
    # latch an opposing override
    self._update(desired=250., measured=150., driver_torque=-(STEER_OVERRIDE_TORQUE_HIGH + 50.),
                 directional_override=True)
    assert self.cc.driver_override

    # torque flips to assisting (same direction as the requested motion) while staying well above
    # _LOW - override must clear immediately, not wait for magnitude to fall below 60
    _, request = self._update(desired=250., measured=150., driver_torque=STEER_OVERRIDE_TORQUE_HIGH + 50.,
                              directional_override=True)
    assert request == 1
    assert not self.cc.driver_override

  def test_magnitude_only_stays_latched_through_direction_flip(self):
    # with the toggle off, a direction flip at sustained high torque must NOT clear the override -
    # legacy magnitude-only hysteresis is unaffected by the directional exit path
    self._update(desired=250., measured=150., driver_torque=-(STEER_OVERRIDE_TORQUE_HIGH + 50.),
                 directional_override=False)
    assert self.cc.driver_override

    _, request = self._update(desired=250., measured=150., driver_torque=STEER_OVERRIDE_TORQUE_HIGH + 50.,
                              directional_override=False)
    assert self.cc.driver_override
    assert request == 0

  def test_yield_resets_on_driver_override(self):
    # engage the angle yield at the limit
    self.cc.apply_steer_last = LKAS_ANGLE_MAX_ACTIVE
    self._update(desired=300., measured=LKAS_ANGLE_MAX_ACTIVE, v_ego=self.LOW_SPEED)
    assert self.cc.lkas_angle_yield

    # driver overrides (opposing torque) while the model still wants the turn and the wheel is still
    # high - this clears the yield (override counts as not lat_active), by design: confirms LKAS
    # strength to the driver
    self._update(desired=300., measured=184., driver_torque=-(STEER_OVERRIDE_TORQUE_HIGH + 50.), v_ego=self.LOW_SPEED)
    assert not self.cc.lkas_angle_yield

    # once the driver releases, active control resumes immediately (snap-back toward the clamped
    # desired angle) rather than continuing to track the measured wheel
    output, request = self._update(desired=300., measured=184., driver_torque=0., v_ego=self.LOW_SPEED)
    assert request == 1
    assert output > 184.


class TestSubaruParams:
  def test_ascent_steer_actuator_delays(self):
    assert CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT).steerActuatorDelay == pytest.approx(0.3)
    assert CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT_2023).steerActuatorDelay == pytest.approx(0.1)
