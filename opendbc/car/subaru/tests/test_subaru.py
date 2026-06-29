import numpy as np
import pytest

from opendbc.car.lateral import apply_std_steer_angle_limits
from opendbc.car.subaru.carcontroller import MAX_ANGLE_TRACKING_ERROR
from opendbc.car.subaru.fingerprints import FW_VERSIONS
from opendbc.car.subaru.interface import CarInterface
from opendbc.car.subaru.values import CAR, CarControllerParams


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
  """Verify the LKAS_ANGLE tracking-error clamp: clamp target before rate-limiting,
  and fall back to measured angle when rate and tracking constraints are mutually impossible."""

  LIMITS = CarControllerParams.ANGLE_LIMITS
  LOW_SPEED = 0.0   # m/s → rate_up = 5 deg/update
  HIGH_SPEED = 20.0  # m/s → rate_up = 0.8 deg/update

  def _rate_up(self, v_ego: float) -> float:
    return float(np.interp(v_ego, self.LIMITS.ANGLE_RATE_LIMIT_UP[0], self.LIMITS.ANGLE_RATE_LIMIT_UP[1]))

  def _apply_clamped(self, model_angle: float, prev: float, measured: float, v_ego: float = 0.) -> float:
    """Replicate the corrected carcontroller logic: clamp target, then rate-limit."""
    desired = float(np.clip(model_angle, measured - MAX_ANGLE_TRACKING_ERROR, measured + MAX_ANGLE_TRACKING_ERROR))
    return apply_std_steer_angle_limits(desired, prev, v_ego, measured, True, self.LIMITS)

  def test_clamp_before_rate_limit_respects_panda_budget(self):
    """Corrected ordering ensures the rate-limited result is within panda's per-frame budget."""
    prev, measured, model_angle = 100., 151., 300.
    result = self._apply_clamped(model_angle, prev, measured)
    assert abs(result - prev) <= self._rate_up(self.LOW_SPEED) + 1e-6

  def test_clamp_before_rate_limit_negative(self):
    prev, measured, model_angle = -100., -151., -300.
    result = self._apply_clamped(model_angle, prev, measured)
    assert abs(result - prev) <= self._rate_up(self.LOW_SPEED) + 1e-6

  def test_fallback_triggered_on_fast_driver_countersteer_positive(self):
    """When rate limit cannot reach the tracking window, the fallback condition is detected."""
    prev, measured, model_angle = 100., 151., 300.
    result = self._apply_clamped(model_angle, prev, measured)
    # Rate limit from 100° can only reach 105°; tracking window minimum is 151°−45°=106° → fallback
    assert abs(result - measured) > MAX_ANGLE_TRACKING_ERROR

  def test_fallback_triggered_on_fast_driver_countersteer_negative(self):
    prev, measured, model_angle = -100., -151., -300.
    result = self._apply_clamped(model_angle, prev, measured)
    assert abs(result - measured) > MAX_ANGLE_TRACKING_ERROR

  def test_no_fallback_during_normal_tight_turn(self):
    """When tracking lag is within the window, result stays there and no fallback fires."""
    prev, measured, model_angle = 100., 90., 200.
    # tracking window: [45, 135]; rate-limited result = 105° → within window
    result = self._apply_clamped(model_angle, prev, measured)
    assert abs(result - measured) <= MAX_ANGLE_TRACKING_ERROR

  def test_inactive_heartbeat_extreme_angle_positive(self):
    """Inactive path returns measured angle even at ±482°, within panda's 545° ceiling."""
    measured = 482.
    result = apply_std_steer_angle_limits(0., 0., self.LOW_SPEED, measured, False, self.LIMITS)
    assert result == pytest.approx(measured)

  def test_inactive_heartbeat_extreme_angle_negative(self):
    measured = -482.
    result = apply_std_steer_angle_limits(0., 0., self.LOW_SPEED, measured, False, self.LIMITS)
    assert result == pytest.approx(measured)

  def test_consecutive_updates_never_exceed_rate(self):
    """Simulate several frames of a tight turn; every consecutive delta respects the rate limit."""
    prev = 0.
    measured = 0.
    rate_up = self._rate_up(self.LOW_SPEED)
    for _ in range(30):
      result = self._apply_clamped(300., prev, measured)
      assert abs(result - prev) <= rate_up + 1e-6
      prev = result
      measured += 5.  # EPS tracking at ~5°/frame


class TestSubaruParams:
  def test_ascent_steer_actuator_delays(self):
    assert CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT).steerActuatorDelay == pytest.approx(0.3)
    assert CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT_2023).steerActuatorDelay == pytest.approx(0.1)
