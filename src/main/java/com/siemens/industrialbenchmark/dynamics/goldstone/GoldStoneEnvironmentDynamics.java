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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

public class GoldStoneEnvironmentDynamics {

	private final int strongestPenaltyAbsIdx;
	private Domain domain;
	private SystemResponse systemResponse;
	private PenaltyFunction currentPenaltyFunction;
	private int phiIdx;
	private final double safeZone;
	private final PenaltyFunction[] penaltyFunctionsArray;
	private static final Logger LOGGER = LoggerFactory.getLogger(GoldStoneEnvironmentDynamics.class);

	public enum Domain {
		POSITIVE(+1),
		INITIAL(0),
		NEGATIVE(-1);

		private final int id;
		Domain(final int id) { this.id = id; }
		public int getValue() { return id; }
		public static Domain fromDouble(final double id) {
			if (id == -1.0) { return NEGATIVE; }
			if (id == 0.0) { return INITIAL; }
			if (id == 1.0) { return POSITIVE; }
			throw new IllegalArgumentException("id must be either [-1, 0, 1], but is " + id);
		}
	}

	public enum SystemResponse {
		ADVANTAGEOUS(+1),
		DISADVANTAGEOUS(-1),
		NEUTRAL(0);

		private final int id;
		SystemResponse(final int id) { this.id = id; }
		public int getValue() { return id; }
		public static SystemResponse fromDouble(final double id) {
			if (id == -1.0) { return DISADVANTAGEOUS; }
			if (id == 0.0) { return NEUTRAL; }
			if (id == 1.0) { return ADVANTAGEOUS; }
			throw new IllegalArgumentException("id must be either [-1, 0, 1], but is " + id);
		}
	}

	public GoldStoneEnvironmentDynamics(final int numberSteps, final double maxRequiredStep, final double safeZone) {
		Preconditions.checkArgument(safeZone >= 0, "safeZone must be non-negative, but is %s.", safeZone);

		this.safeZone = safeZone;
		this.strongestPenaltyAbsIdx = computeStrongestPenaltyAbsIdx(numberSteps);
		this.penaltyFunctionsArray = defineRewardFunctions(numberSteps, maxRequiredStep);
		reset();
	}

	private void reset() {
		domain = Domain.INITIAL;
		phiIdx = 0;
		systemResponse = SystemResponse.ADVANTAGEOUS;
		currentPenaltyFunction = getPenaltyFunction();
	}

	public double rewardAt(final double pos) {
		return -currentPenaltyFunction.reward(pos);
	}

	public double optimalPosition() {
		return currentPenaltyFunction.getOptimumRadius();
	}

	public double optimalReward() {
		return -currentPenaltyFunction.getOptimumValue();
	}

	public void stateTransition(final double newControlValue) {

		final Domain oldDomain = domain;

		// (0) compute new domain
		domain = computeDomain(newControlValue);

		// (1) if domain change: system response <- advantageous
		if (domain != oldDomain) { // FIXME use equals instead, and implement it for Domain (also hashCode())
			systemResponse = SystemResponse.ADVANTAGEOUS;
			LOGGER.trace("  turning sys behavior -> advantageous");
		}

		// (2) compute & apply turn direction
		phiIdx += computeAngularStep(newControlValue);

		// (3) update system response if necessary
		systemResponse = updateSystemResponse(phiIdx);

		// (4) if Phi_index == 0: reset internal state
		if (phiIdx == 0 && Math.abs(newControlValue) <= safeZone) {
			reset();
		}

		// (5) apply symmetry
		phiIdx = applySymmetry(phiIdx);

		LOGGER.trace("  phiIdx = " + phiIdx);
		currentPenaltyFunction = getPenaltyFunction();
	}

	/**
	 * Computes the new domain of control action.
	 * Note:
	 * if control action is in safe zone, domain remains unchanged
	 * as 'penalty landscape' turn direction is independent of exact position
	 * in safe zone, reset to Domain.initial can be applied later.
	 *
	 * @param newPosition The new position.
	 * @return The numerical value of the new domain.
	 */
	private Domain computeDomain(final double newPosition) {
		if (Math.abs(newPosition) <= safeZone) {
			return domain;
		} else {
			return Domain.fromDouble(Math.signum(newPosition));
		}
	}


	private double computeAngularStep(final double newPosition) {
		// cool down when position close to zero
		if (Math.abs(newPosition) <= safeZone) {
			return -Math.signum(phiIdx);
		}

		if (phiIdx == (-domain.getValue() * strongestPenaltyAbsIdx)) {
			LOGGER.trace("  no turning");
			return 0;
		}

		return systemResponse.getValue() * Math.signum(newPosition);
	}

	/**
	 * Only changes the system response if the turn angle hits 90deg
	 * in the domain of the current position.
	 * I.e.:
	 * <code>new_position &gt;  this.__safe_zone and new_Phi_idx =  90deg</code>
	 * <code>new_position &lt; -this.__safe_zone and new_Phi_idx = -90deg</code>
	 *
	 * @param phiIdx
	 * @return the resulting system response
	 */
	private SystemResponse updateSystemResponse(final int newPhiIdx) {
		if (Math.abs(newPhiIdx) >= strongestPenaltyAbsIdx) {
			LOGGER.trace("  turning sys behavior -> disadvantageous");
			return SystemResponse.DISADVANTAGEOUS;
		} else {
			return this.systemResponse;
		}
	}

	/**
	 * Calculates a symmetric version of a given phi.
	 * By employing reflection symmetry with respect to the x-axis:
	 * <ul>
	 *	<li>Phi -&gt; pi -Phi</li>
	 *	<li>turn direction -&gt; turn direction</li>
	 *	<li>x -&gt; x</li>
	 * </ul>
	 * moves 'penalty landscape' rotation angle in domain
	 * <code>[-90deg ... +90deg]</code>
	 * corresponding to <code>Phi_index</code> in
	 * <code>[ -strongest_penality_abs_idx*angular_speed, ..., -angular_speed, 0, angular_speed, ..., strongest_penality_abs_idx*angular_speed ]</code>
	 *
	 * @param otherPhiIdx
	 * @return the symmetric phi
	 */
	private int applySymmetry(final int otherPhiIdx) {

		int currentPhiIdx = otherPhiIdx;

		/*
		 * Do nothing if 'penalty landscape' rotation angle is in
		 * <math>[-90deg ... +90deg]</math>
		 * corresponding to angle indices
		 * <math>[-self.__strongest_penality_abs_idx, ...-1,0,1, ..., self.__strongest_penality_abs_idx-]</math>
		 */
		if (Math.abs(currentPhiIdx) <= strongestPenaltyAbsIdx) {
			return currentPhiIdx;
		}

		/*
		 * Otherwise:
		 * Use 2pi symmetry to move angle index p in domain
		 * [0 ... 360deg)
		 * corresponding to angle indices
		 * [0, ..., 4*self.__strongest_penality_abs_idx-1]
		 * But we are only executing the following code, if the angle is in
		 * (90deg, ..., 270deg)
		 * corresponding to angle indices
		 * [self.__strongest_penality_abs_idx+1, ..., 3*self.__strongest_penality_abs_idx-1]
		 * Therefore, the reflection-symmetry operation
		 * p <- 2*self.__strongest_penality_abs_idx - p
		 * will transform p back into the desired angle indices domain
		 * [-self.__strongest_penality_abs_idx, ...-1,0,1, ..., self.__strongest_penality_abs_idx-]
		 * */
		currentPhiIdx = currentPhiIdx % (4 * strongestPenaltyAbsIdx);
		if (currentPhiIdx < 0) {
			currentPhiIdx += (4 * strongestPenaltyAbsIdx);
		}
		currentPhiIdx = 2 * strongestPenaltyAbsIdx - currentPhiIdx;

		return currentPhiIdx;
	}

	public PenaltyFunction getPenaltyFunction() {
		return getPenaltyFunction(phiIdx);
	}

	public PenaltyFunction getPenaltyFunction(final int otherPhiIdx) {
		int idx = strongestPenaltyAbsIdx + this.applySymmetry(otherPhiIdx);
		if (idx < 0) {
			idx += penaltyFunctionsArray.length;
		}
		return penaltyFunctionsArray[idx];
	}

	/**
	 * @param numberSteps
	 *   the number of steps required for one full cycle of the optimal policy.
	 *   For easy numerics, it is required that this is positive and an integer
	 *   multiple of 4.
	 *   By employing reflection symmetry with respect to x-axis:
	 *   <ul>
	 *		<li>Phi -&gt; pi -Phi</li>
	 *		<li>turn direction -&gt; turn direction</li>
	 *		<li>x -&gt; x</li>
	 *   </ul>
	 *   The required rewards functions can be restricted to turn angles Phi in
	 *   <code>[-90deg ... +90deg]</code> of the 'penalty landscape'.
	 *   One quarter-segment (e.g <code>[0 ... 90deg]</code>) of the entire
	 *   'penalty landscape' is divided into <code>numberSteps / 4</code> steps.
	 *   Note that <code>numberSteps / 4</code> is an integer per requirement
	 *   from above.
	 * @param maxRequiredStep
	 *
	 * Implementation:
	 * According to the above explanation, the 'penalty landscape' turns
	 * <code>angular_speed = 360deg / numberSteps</code>
	 * in each state transition, or does not turn at all.
	 * Per convention, the 'penalty landscape' positions are confined to a
	 * homogeneously spaced grid of turn angles
	 * <code>[ 0, angular_speed, 2*angular_speed, ... , (numberSteps -1)*angular_speed ]</code>.
	 * Lets define <code>strongest_penality_abs_idx = numberSteps / 4</code>.
	 * Hence,
	 * <code>strongest_penality_abs_idx*angular_speed = (numberSteps / 4) * (360deg /  numberSteps)= 90deg</code>.
	 * It follows that:
	 *   <ul>
	 *		<li><code>Phi = - strongest_penality_abs_idx*angular_speed</code>
	 *			maximized the control penalties for the positive domain
	 *			(i.e. where <code>x &gt; 0</code>)</li>
	 *		<li><code>Phi = + strongest_penality_abs_idx*angular_speed</code>
	 *			maximized the control penalties for the negative domain
	 *			(i.e. where <code>x &lt; 0</code>)</li>
	 *   </ul>
	 * Exploiting reflection symmetry,
	 * the required grid of reward functions can be reduced to
	 * <code>[ -strongest_penality_abs_idx*angular_speed, ..., -angular_speed, 0, angular_speed, ..., strongest_penality_abs_idx*angular_speed ]</code>
	 * This has the advantage, that either end of the grid represents
	 * the worst case points of the ???.
	 */
	private static PenaltyFunction[] defineRewardFunctions(final int numberSteps, final double maxRequiredStep) {

		final int k = computeStrongestPenaltyAbsIdx(numberSteps);
		final double[] angleGid = new double[k * 2 + 1];
		for (int i = -k; i <= k; i++) {
			angleGid[i + k] = i * 2 * Math.PI / numberSteps;
		}
		final PenaltyFunction[] penaltyFunctionsArray = new PenaltyFunction[angleGid.length];
		for (int i = 0; i < angleGid.length; i++) {
			penaltyFunctionsArray[i] = new PenaltyFunction(angleGid[i], maxRequiredStep);
		}

		return penaltyFunctionsArray;
	}

	private static int computeStrongestPenaltyAbsIdx(final int numberSteps) {
		Preconditions.checkArgument(numberSteps >= 1 && (numberSteps %4) == 0,
				"numberSteps must be positive and an integer multiple of 4, but is %s", numberSteps);

		return numberSteps / 4;
	}

	public Domain getDomain() {
		return domain;
	}

	public void setDomain(final Domain domain) {
		this.domain = domain;
	}

	public SystemResponse getSystemResponse() {
		return systemResponse;
	}

	public void setSystemResponse(final SystemResponse systemResponse) {
		this.systemResponse = systemResponse;
	}

	public int getPhiIdx() {
		return phiIdx;
	}

	public void setPhiIdx(final int phiIdx) {
		this.phiIdx = phiIdx;
	}

	protected int getStrongestPenaltyAbsIdx() {
		return strongestPenaltyAbsIdx;
	}

	protected PenaltyFunction getCurrentPenaltyFunction() {
		return currentPenaltyFunction;
	}

	protected double getSafeZone() {
		return safeZone;
	}

	protected PenaltyFunction[] getPenaltyFunctionsArray() {
		return penaltyFunctionsArray;
	}
}
