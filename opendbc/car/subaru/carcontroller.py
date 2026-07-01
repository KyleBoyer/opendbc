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

# Modern Subaru angle-LKAS EPS units hard-fault when an ACTIVE steering request reaches ~200 deg
# (an independent tester confirmed 199 deg works, 200 deg faults; it is the requested LKAS_Output
# that matters, not the measured wheel). This was first seen below 10 mph, but it also faults just
# above 10 mph (observed at 11.2 mph), and a hard speed gate created a discontinuity: the clamp
# released as speed crossed the gate, jerking the command up past 200 deg into the fault. So the
# limit is enforced at ALL speeds (>=195 deg is only ever reachable in slow, sharp turns anyway).
# Clamp the active request just under the limit so the wheel still steers to near full lock while
# keeping LKAS_Request set (no request-bit chatter); only once the measured wheel itself reaches the
# limit do we drop the request and track measured (heartbeat). Hysteresis on the release avoids chatter.
LKAS_ANGLE_MAX_ACTIVE = 195.0    # deg; max active request magnitude
LKAS_ANGLE_YIELD_RELEASE = 185.0  # deg; measured must fall below this to resume active control

# Driver override: when the driver applies steering torque, stop actively requesting (track measured)
# so they can steer freely - e.g. ease out of a turn the model still wants to hold. Hysteresis (release
# below _LOW) prevents request chatter. Normal angle-LKAS steering keeps driver torque under ~80 (the
# steeringPressed threshold), so STEER_OVERRIDE_TORQUE_HIGH is clearly a deliberate driver input.
STEER_OVERRIDE_TORQUE_HIGH = 100  # enter override
STEER_OVERRIDE_TORQUE_LOW = 60    # exit override

# When directional override is enabled (CC_SP.subaruDirectionalSteerOverride), only treat torque that
# opposes the requested motion (desired - measured, i.e. which way we're actually trying to move the
# wheel from here) as an override - pushing hard in that same direction (helping) no longer drops the
# active request. Comparing against raw desired angle instead of the delta would misclassify e.g.
# unwind assistance: desired=100/measured=150 (unwinding toward 100) with the driver helping via
# negative torque is genuinely assisting, even though desired_angle itself is still positive. Below
# this much requested motion, direction is unreliable (steady-state tracking / straight-ahead noise),
# so fall back to magnitude-only gating.
STEER_OVERRIDE_ANGLE_SIGN_FLOOR = 5.0  # deg


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP, CP_SP):
    super().__init__(dbc_names, CP, CP_SP)
    self.apply_torque_last = 0
    self.apply_steer_last = 0

    self.cruise_button_prev = 0
    self.steer_rate_counter = 0

    self.lkas_angle_yield = False
    self.driver_override = False

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
        # Driver override: once the driver applies steering torque, stop actively requesting so they
        # can steer freely (e.g. ease out of a turn the model still wants to hold). Hysteresis.
        # When directional override is on, only torque that opposes the requested motion counts -
        # pushing hard the same way we're already trying to move (helping) never latches an override,
        # and if the driver flips from opposing to helping mid-override it clears immediately rather
        # than waiting for torque to fall below _LOW. Direction-agnostic torque-magnitude exit still
        # applies too, so a driver who just relaxes (regardless of direction) always regains control.
        abs_driver_torque = abs(CS.out.steeringTorque)
        requested_motion = actuators.steeringAngleDeg - CS.out.steeringAngleDeg
        opposing = (not CC_SP.subaruDirectionalSteerOverride or abs(requested_motion) <= STEER_OVERRIDE_ANGLE_SIGN_FLOOR
                    or CS.out.steeringTorque * requested_motion < 0)
        if abs_driver_torque > STEER_OVERRIDE_TORQUE_HIGH and opposing:
          self.driver_override = True
        elif abs_driver_torque < STEER_OVERRIDE_TORQUE_LOW or not opposing:
          self.driver_override = False
        lat_active = CC.latActive and not self.driver_override

        apply_steer = apply_std_steer_angle_limits(actuators.steeringAngleDeg, self.apply_steer_last, CS.out.vEgoRaw,
                                                   CS.out.steeringAngleDeg, lat_active, CarControllerParams.ANGLE_LIMITS)
        apply_steer_req = lat_active

        # Never actively request an angle the EPS will fault on (enforced at all speeds to avoid a
        # speed-gate discontinuity). Clamping rather than dropping the request lets the wheel keep
        # steering to the limit without request-bit chatter.
        if lat_active:
          apply_steer = float(np.clip(apply_steer, -LKAS_ANGLE_MAX_ACTIVE, LKAS_ANGLE_MAX_ACTIVE))

        # Once the measured wheel reaches the limit, stop actively requesting and track the measured
        # angle (heartbeat) so we don't fight the wheel at full lock. Hold the yield until BOTH the
        # model request and the measured wheel fall below the release angle: releasing while the model
        # still wants a large angle re-engages an active command that pushes against the (overshooting
        # or exiting) wheel, which jerks the steering. Releasing only when the turn is genuinely
        # ending lets the command track the wheel back down smoothly. The yield resets on driver
        # override clearing too (lat_active, not CC.latActive) so the system resumes tracking as soon
        # as the driver lets go - the resulting snap-back is intentional, confirming LKAS strength.
        if not lat_active:
          self.lkas_angle_yield = False
        elif not self.lkas_angle_yield:
          if abs(CS.out.steeringAngleDeg) >= LKAS_ANGLE_MAX_ACTIVE:
            self.lkas_angle_yield = True
        elif max(abs(actuators.steeringAngleDeg), abs(CS.out.steeringAngleDeg)) < LKAS_ANGLE_YIELD_RELEASE:
          self.lkas_angle_yield = False

        if not lat_active or self.lkas_angle_yield:
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
