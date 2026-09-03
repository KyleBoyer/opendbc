import pytest

from opendbc.can import CANDefine
from opendbc.car import Bus, structs
from opendbc.car.subaru.carstate import CarState
from opendbc.car.subaru.values import CAR, DBC


class TestSubaruGearParsing:
  @pytest.mark.parametrize("ratio", ("1", "2", "3", "4", "5", "6", "7", "8"))
  def test_manual_ratio_is_manumatic(self, ratio):
    assert CarState.parse_gear_shifter(ratio) == structs.CarState.GearShifter.manumatic

  @pytest.mark.parametrize(("raw_gear", "ratio"), (
    (137, "1"), (145, "2"), (153, "3"), (161, "4"),
    (169, "5"), (177, "6"), (185, "7"), (193, "8"),
  ))
  def test_global_manual_ratio_can_values(self, raw_gear, ratio):
    can_define = CANDefine(DBC[CAR.SUBARU_ASCENT_2023][Bus.pt])
    gear = can_define.dv["Transmission"]["Gear"][raw_gear]
    assert gear == ratio
    assert CarState.parse_gear_shifter(gear) == structs.CarState.GearShifter.manumatic

  @pytest.mark.parametrize(("gear", "expected"), (
    ("D", structs.CarState.GearShifter.drive),
    ("N", structs.CarState.GearShifter.neutral),
    ("R", structs.CarState.GearShifter.reverse),
    ("P", structs.CarState.GearShifter.park),
    ("9", structs.CarState.GearShifter.unknown),
    (None, structs.CarState.GearShifter.unknown),
  ))
  def test_non_manual_gear_uses_common_parser(self, gear, expected):
    assert CarState.parse_gear_shifter(gear) == expected
