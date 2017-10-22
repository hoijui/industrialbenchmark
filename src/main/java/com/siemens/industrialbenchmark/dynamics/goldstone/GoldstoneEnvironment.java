/*
Copyright 2016 Siemens AG.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package com.siemens.industrialbenchmark.dynamics.goldstone;

import com.siemens.industrialbenchmark.dynamics.goldstone.GoldStoneEnvironmentDynamics.Domain;
import com.siemens.industrialbenchmark.dynamics.goldstone.GoldStoneEnvironmentDynamics.SystemResponse;

public class GoldstoneEnvironment {

	private static final double DEFAULT_CONTROL_START_VALUE = 0.0;

	private final GoldStoneEnvironmentDynamics dynamics;
	private double controlPosition;

	public GoldstoneEnvironment(final int numberSteps, final double maxRequiredStep, final double safeZone) {
		this.dynamics = new GoldStoneEnvironmentDynamics(numberSteps, maxRequiredStep, safeZone);
		this.controlPosition = DEFAULT_CONTROL_START_VALUE;
	}

	// for eventual serialization
	protected GoldStoneEnvironmentDynamics getDynamics() {
		return dynamics;
	}

	public void reset() {
		setControlStartValue(DEFAULT_CONTROL_START_VALUE);
	}

	public void setControlStartValue(final double controlStartValue) {
		this.controlPosition = controlStartValue;
	}

	public double reward() {
		return dynamics.rewardAt(controlPosition);
	}

	public double optimalPosition() {
		return dynamics.optimalPosition();
	}

	public double optimalReward() {
		return dynamics.optimalReward();
	}

	/**
	 * Applies action and returns reward.
	 * @param controlValueChange the relative change/delta to apply to the control value
	 * @return the reward
	 */
	public double stateTransition(final double controlValueChange) {
		// TODO: (comment from Alex) not yet implemented: test if action is allowed
		controlPosition += controlValueChange;
		dynamics.stateTransition(controlPosition);
		return reward();
	}

	public PenaltyFunction getRewardFunction() {
		return dynamics.getPenaltyFunction();
	}

	public double getControlPosition() {
		return controlPosition;
	}

	public void setControlPosition(final double controlPosition) {
		this.controlPosition = controlPosition;
		dynamics.stateTransition(controlPosition);
		reward();
	}

	public float getDomain() {
		return dynamics.getDomain().getValue();
	}

	public void setDomain(final double double1) {
		dynamics.setDomain(Domain.fromDouble(double1));
	}

	public float getSystemResponse() {
		return dynamics.getSystemResponse().getValue();
	}

	public void setSystemResponse(final double systemResponse) {
		dynamics.setSystemResponse(SystemResponse.fromDouble(systemResponse));
	}

	public float getPhiIdx() {
		return dynamics.getPhiIdx();
	}

	public void setPhiIdx(final double phiIdx) {
		dynamics.setPhiIdx((int) phiIdx);
	}
}

