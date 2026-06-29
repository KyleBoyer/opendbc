import pytest

from opendbc.car.lateral import apply_std_steer_angle_limits
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


class TestSubaruParams:
  def test_ascent_steer_actuator_delays(self):
    assert CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT).steerActuatorDelay == pytest.approx(0.3)
    assert CarInterface.get_non_essential_params(CAR.SUBARU_ASCENT_2023).steerActuatorDelay == pytest.approx(0.1)
