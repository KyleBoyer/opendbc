import numpy as np
from opendbc.can import CANPacker
from opendbc.car import Bus, make_tester_present_msg
from opendbc.car.lateral import apply_driver_steer_torque_limits, apply_std_steer_angle_limits, common_fault_avoidance
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.subaru import subarucan
from opendbc.car.subaru.values import DBC, GLOBAL_ES_ADDR, CanBus, CarControllerParams, SubaruFlags

# FIXME: These limits aren't exact. The real limit is more than likely over a larger time period and
# involves the total steering angle change rather than rate, but these limits work well for now
MAX_STEER_RATE = 25  # deg/s
MAX_STEER_RATE_FRAMES = 7  # tx control frames needed before torque can be cut

# LKAS_ANGLE EPS effort (torque) inhibitor.
# The LKAS_ANGLE EPS latches a permanent Steer_Error_1 when its self-drive torque output stays near
# its ceiling (~2250 units) for too long. This happens on tight low-speed turns the wheel cannot
# physically follow, where the EPS hits its effort limit. It is NOT a tracking-error or thermal
# limit: the same EPS delivers ~9800 units under driver-assisted steering without faulting, and
# clamping the command-to-measured tracking error (tried 45° and 20°) did not prevent the fault.
# Log analysis of two faulting turns: normal active driving peaks ~620 units, while both faults
# sustained ~1400 units for ~1.16s before latching. So inhibit the angle request once EPS torque is
# sustained above EPS_TORQUE_HIGH, anchor the command to the measured angle, and hold the cutout
# with hysteresis until torque relaxes below EPS_TORQUE_RELEASE. This is a stateful cutout, not a
# per-frame proportional feedback: the EPS torque signal is delayed and noisy enough that direct
# feedback could oscillate.
EPS_TORQUE_HIGH = 1400         # units; sustained EPS torque output above this starts inhibition
EPS_TORQUE_RELEASE = 1000      # units; EPS torque must fall back below this to release
EPS_TORQUE_FAULT_FRAMES = 12   # ~0.24s at 50Hz of sustained high torque before inhibiting
EPS_INHIBIT_MIN_FRAMES = 20    # hold inhibition at least ~0.4s
EPS_RELEASE_FRAMES = 10        # require ~0.2s below the release threshold before re-engaging


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP, CP_SP):
    super().__init__(dbc_names, CP, CP_SP)
    self.apply_torque_last = 0
    self.apply_steer_last = 0

    self.cruise_button_prev = 0
    self.steer_rate_counter = 0

    # LKAS_ANGLE EPS effort inhibitor state
    self.eps_inhibit = False
    self.eps_high_frames = 0
    self.eps_low_frames = 0
    self.eps_inhibit_frames = 0

    self.p = CarControllerParams(CP)
    self.packer = CANPacker(DBC[CP.carFingerprint][Bus.pt])

  def update(self, CC, CC_SP, CS, now_nanos):
    actuators = CC.actuators
    hud_control = CC.hudControl
    pcm_cancel_cmd = CC.cruiseControl.cancel

    can_sends = []

    # *** steering ***
    if (self.frame % self.p.STEER_STEP) == 0:
      apply_steer = 0
      apply_torque = 0
      if self.CP.flags & SubaruFlags.LKAS_ANGLE:
        apply_steer = apply_std_steer_angle_limits(actuators.steeringAngleDeg, self.apply_steer_last, CS.out.vEgoRaw,
                                                   CS.out.steeringAngleDeg, CC.latActive, CarControllerParams.ANGLE_LIMITS)
        apply_steer_req = CC.latActive

        # EPS effort inhibitor: when the EPS self-drive torque is sustained near its fault ceiling
        # (tight low-speed turn the wheel can't follow), cut the request and anchor to the measured
        # angle so the EPS isn't driven into a permanent Steer_Error_1. Hysteretic stateful cutout.
        eps_torque = abs(CS.out.steeringTorqueEps)
        if not CC.latActive:
          self.eps_inhibit = False
          self.eps_high_frames = 0
        elif not self.eps_inhibit:
          self.eps_high_frames = self.eps_high_frames + 1 if eps_torque > EPS_TORQUE_HIGH else 0
          if self.eps_high_frames >= EPS_TORQUE_FAULT_FRAMES:
            self.eps_inhibit = True
            self.eps_inhibit_frames = 0
            self.eps_low_frames = 0
        else:
          self.eps_inhibit_frames += 1
          self.eps_low_frames = self.eps_low_frames + 1 if eps_torque < EPS_TORQUE_RELEASE else 0
          if self.eps_inhibit_frames >= EPS_INHIBIT_MIN_FRAMES and self.eps_low_frames >= EPS_RELEASE_FRAMES:
            self.eps_inhibit = False
            self.eps_high_frames = 0

        if not CC.latActive or self.eps_inhibit:
          # anchor to measured angle and drop the request (inactive, or EPS effort inhibited)
          apply_steer = CS.out.steeringAngleDeg
          apply_steer_req = False

        can_sends.append(subarucan.create_steering_control_angle(self.packer, apply_steer, apply_steer_req))
        self.apply_steer_last = apply_steer

      # torque-based steering
      else:
        apply_torque = int(round(actuators.torque * self.p.STEER_MAX))

        # limits due to driver torque

        new_torque = int(round(apply_torque))
        apply_torque = apply_driver_steer_torque_limits(new_torque, self.apply_torque_last, CS.out.steeringTorque, self.p)

        if not CC.latActive:
          apply_torque = 0

        if self.CP.flags & SubaruFlags.PREGLOBAL:
          can_sends.append(subarucan.create_preglobal_steering_control(self.packer, self.frame // self.p.STEER_STEP, apply_torque, CC.latActive))
        else:
          apply_steer_req = CC.latActive

          if self.CP.flags & SubaruFlags.STEER_RATE_LIMITED:
            # Steering rate fault prevention
            self.steer_rate_counter, apply_steer_req = \
              common_fault_avoidance(abs(CS.out.steeringRateDeg) > MAX_STEER_RATE, apply_steer_req,
                                    self.steer_rate_counter, MAX_STEER_RATE_FRAMES)

          can_sends.append(subarucan.create_steering_control(self.packer, apply_torque, apply_steer_req))

        self.apply_torque_last = apply_torque

    # *** longitudinal ***

    if CC.longActive:
      apply_throttle = int(round(np.interp(actuators.accel, CarControllerParams.THROTTLE_LOOKUP_BP, CarControllerParams.THROTTLE_LOOKUP_V)))
      apply_rpm = int(round(np.interp(actuators.accel, CarControllerParams.RPM_LOOKUP_BP, CarControllerParams.RPM_LOOKUP_V)))
      apply_brake = int(round(np.interp(actuators.accel, CarControllerParams.BRAKE_LOOKUP_BP, CarControllerParams.BRAKE_LOOKUP_V)))

      # limit min and max values
      cruise_throttle = np.clip(apply_throttle, CarControllerParams.THROTTLE_MIN, CarControllerParams.THROTTLE_MAX)
      cruise_rpm = np.clip(apply_rpm, CarControllerParams.RPM_MIN, CarControllerParams.RPM_MAX)
      cruise_brake = np.clip(apply_brake, CarControllerParams.BRAKE_MIN, CarControllerParams.BRAKE_MAX)
    else:
      cruise_throttle = CarControllerParams.THROTTLE_INACTIVE
      cruise_rpm = CarControllerParams.RPM_MIN
      cruise_brake = CarControllerParams.BRAKE_MIN

    # *** alerts and pcm cancel ***
    if self.CP.flags & SubaruFlags.PREGLOBAL:
      if self.frame % 5 == 0:
        # 1 = main, 2 = set shallow, 3 = set deep, 4 = resume shallow, 5 = resume deep
        # disengage ACC when OP is disengaged
        if pcm_cancel_cmd:
          cruise_button = 1
        # turn main on if off and past start-up state
        elif not CS.out.cruiseState.available and CS.ready:
          cruise_button = 1
        else:
          cruise_button = CS.cruise_button

        # unstick previous mocked button press
        if cruise_button == 1 and self.cruise_button_prev == 1:
          cruise_button = 0
        self.cruise_button_prev = cruise_button

        can_sends.append(subarucan.create_preglobal_es_distance(self.packer, cruise_button, CS.es_distance_msg))

    else:
      if self.frame % 10 == 0:
        can_sends.append(subarucan.create_es_dashstatus(self.packer, self.frame // 10, CS.es_dashstatus_msg, CC.enabled,
                                                        self.CP.openpilotLongitudinalControl, CC.longActive, hud_control.leadVisible))

        can_sends.append(subarucan.create_es_lkas_state(self.packer, self.frame // 10, CS.es_lkas_state_msg, CC.enabled, hud_control.visualAlert,
                                                        hud_control.leftLaneVisible, hud_control.rightLaneVisible,
                                                        hud_control.leftLaneDepart, hud_control.rightLaneDepart))

        if self.CP.flags & SubaruFlags.SEND_INFOTAINMENT:
          can_sends.append(subarucan.create_es_infotainment(self.packer, self.frame // 10, CS.es_infotainment_msg, hud_control.visualAlert))

      if self.CP.openpilotLongitudinalControl:
        if self.frame % 5 == 0:
          can_sends.append(subarucan.create_es_status(self.packer, self.frame // 5, CS.es_status_msg,
                                                      self.CP.openpilotLongitudinalControl, CC.longActive, cruise_rpm))

          can_sends.append(subarucan.create_es_brake(self.packer, self.frame // 5, CS.es_brake_msg,
                                                     self.CP.openpilotLongitudinalControl, CC.longActive, cruise_brake))

          can_sends.append(subarucan.create_es_distance(self.packer, self.frame // 5, CS.es_distance_msg, 0, pcm_cancel_cmd,
                                                        self.CP.openpilotLongitudinalControl, cruise_brake > 0, cruise_throttle))
      else:
        if pcm_cancel_cmd:
          if not (self.CP.flags & SubaruFlags.HYBRID):
            bus = CanBus.alt if self.CP.flags & SubaruFlags.GLOBAL_GEN2 else CanBus.main
            can_sends.append(subarucan.create_es_distance(self.packer, CS.es_distance_msg["COUNTER"] + 1, CS.es_distance_msg, bus, pcm_cancel_cmd))

      if self.CP.flags & SubaruFlags.DISABLE_EYESIGHT:
        # Tester present (keeps eyesight disabled)
        if self.frame % 100 == 0:
          can_sends.append(make_tester_present_msg(GLOBAL_ES_ADDR, CanBus.camera, suppress_response=True))

        # Create all of the other eyesight messages to keep the rest of the car happy when eyesight is disabled
        if self.frame % 5 == 0:
          can_sends.append(subarucan.create_es_highbeamassist(self.packer))

        if self.frame % 10 == 0:
          can_sends.append(subarucan.create_es_static_1(self.packer))

        if self.frame % 2 == 0:
          can_sends.append(subarucan.create_es_static_2(self.packer))

    new_actuators = actuators.as_builder()
    if self.CP.flags & SubaruFlags.LKAS_ANGLE:
      new_actuators.steeringAngleDeg = self.apply_steer_last
    else:
      new_actuators.torque = self.apply_torque_last / self.p.STEER_MAX
      new_actuators.torqueOutputCan = self.apply_torque_last

    self.frame += 1
    return new_actuators, can_sends
