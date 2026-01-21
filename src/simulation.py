#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Simulation Engine
================================================================================

Project:        Week 1 Project 1: The Audible Integrator
Module:         simulation.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 21, 2026
Last Updated:   January 21, 2026

License:        MIT License
================================================================================

This module ties together the integrators, physics, and sonification
to run complete simulations with real-time audio feedback.
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time

from .integrators import (
    NumericalIntegrator,
    ForwardEuler,
    RungeKutta4,
    VelocityVerlet,
    Leapfrog,
    get_integrator
)
from .physics import PhysicalSystem, HarmonicOscillator, PhysicalState
from .sonification import (
    create_audio_engine,
    SonificationConfig,
    AUDIO_AVAILABLE
)


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    dt: float = 0.01              # Time step (seconds)
    duration: float = 10.0         # Total simulation time
    realtime: bool = True          # Run in real-time (with audio)
    audio_enabled: bool = True     # Enable audio sonification
    store_history: bool = True     # Store full trajectory


@dataclass 
class SimulationResult:
    """Results from a simulation run."""
    integrator_name: str
    is_symplectic: bool
    
    # Time series data
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    energies: np.ndarray
    
    # Energy statistics
    initial_energy: float
    final_energy: float
    energy_drift: float           # (E_final - E_initial) / E_initial
    energy_std: float             # Standard deviation of energy
    max_energy_deviation: float   # Maximum |E - E₀| / E₀
    
    # Audio parameters (if tracked)
    frequencies: Optional[np.ndarray] = None
    amplitudes: Optional[np.ndarray] = None
    distortions: Optional[np.ndarray] = None


class Simulator:
    """
    Main simulation engine.
    
    Combines physics, integration, and sonification into a unified
    simulation framework.
    """
    
    def __init__(
        self,
        system: PhysicalSystem,
        integrator: NumericalIntegrator,
        config: Optional[SimulationConfig] = None,
        audio_config: Optional[SonificationConfig] = None
    ):
        self.system = system
        self.integrator = integrator
        self.config = config or SimulationConfig()
        self.audio_config = audio_config or SonificationConfig()
        
        # State
        self.position: Optional[np.ndarray] = None
        self.velocity: Optional[np.ndarray] = None
        self.time: float = 0.0
        
        # History
        self.history: List[PhysicalState] = []
        
        # Audio engine
        self.audio_engine = None
    
    def initialize(
        self,
        position: np.ndarray,
        velocity: np.ndarray
    ):
        """Initialize the simulation with starting conditions."""
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.time = 0.0
        self.history = []
        
        # Reset integrator state if needed
        if hasattr(self.integrator, 'reset'):
            self.integrator.reset()
        
        # Store initial state
        if self.config.store_history:
            self.history.append(self.system.get_state(
                self.time, self.position, self.velocity
            ))
    
    def step(self) -> PhysicalState:
        """Perform one integration step."""
        # Get acceleration function
        if hasattr(self.system, '_compute_acceleration'):
            # Double pendulum needs velocity too
            accel_func = lambda pos: self.system._compute_acceleration(pos, self.velocity)
        else:
            accel_func = self.system.acceleration
        
        # Integrate
        self.position, self.velocity = self.integrator.step(
            self.position,
            self.velocity,
            accel_func,
            self.config.dt
        )
        self.time += self.config.dt
        
        # Get new state
        state = self.system.get_state(self.time, self.position, self.velocity)
        
        # Store history
        if self.config.store_history:
            self.history.append(state)
        
        return state
    
    def run(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        progress_callback: Optional[Callable[[float, PhysicalState], None]] = None
    ) -> SimulationResult:
        """
        Run a complete simulation.
        
        Args:
            position: Initial position
            velocity: Initial velocity
            progress_callback: Optional callback(time, state) for updates
            
        Returns:
            SimulationResult with full trajectory and statistics
        """
        self.initialize(position, velocity)
        
        num_steps = int(self.config.duration / self.config.dt)
        initial_energy = self.system.total_energy(position, velocity)
        
        # Set up audio if enabled
        if self.config.audio_enabled and self.config.realtime:
            self.audio_engine = create_audio_engine(self.audio_config)
            self.audio_engine.initialize(initial_energy)
            self.audio_engine.start()
        
        # Timing for realtime mode
        start_real_time = time.time()
        
        try:
            for i in range(num_steps):
                state = self.step()
                
                # Update audio
                if self.audio_engine:
                    self.audio_engine.update_energy(state.total_energy)
                
                # Progress callback
                if progress_callback:
                    progress_callback(self.time, state)
                
                # Realtime pacing
                if self.config.realtime:
                    target_real_time = start_real_time + self.time
                    current_real_time = time.time()
                    sleep_time = target_real_time - current_real_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        finally:
            # Clean up audio
            if self.audio_engine:
                self.audio_engine.stop()
        
        return self._compile_results()
    
    def run_fast(
        self,
        position: np.ndarray,
        velocity: np.ndarray
    ) -> SimulationResult:
        """Run simulation as fast as possible (no realtime, no audio)."""
        saved_realtime = self.config.realtime
        saved_audio = self.config.audio_enabled
        
        self.config.realtime = False
        self.config.audio_enabled = False
        
        try:
            return self.run(position, velocity)
        finally:
            self.config.realtime = saved_realtime
            self.config.audio_enabled = saved_audio
    
    def _compile_results(self) -> SimulationResult:
        """Compile simulation history into results."""
        times = np.array([s.time for s in self.history])
        positions = np.array([s.position for s in self.history])
        velocities = np.array([s.velocity for s in self.history])
        energies = np.array([s.total_energy for s in self.history])
        
        initial_energy = energies[0]
        final_energy = energies[-1]
        
        result = SimulationResult(
            integrator_name=self.integrator.name,
            is_symplectic=self.integrator.is_symplectic,
            times=times,
            positions=positions,
            velocities=velocities,
            energies=energies,
            initial_energy=initial_energy,
            final_energy=final_energy,
            energy_drift=(final_energy - initial_energy) / initial_energy,
            energy_std=np.std(energies),
            max_energy_deviation=np.max(np.abs(energies - initial_energy)) / initial_energy
        )
        
        # Add audio tracking if available
        if self.audio_engine and hasattr(self.audio_engine, 'frequency_history'):
            result.frequencies = np.array(self.audio_engine.frequency_history)
            result.amplitudes = np.array(self.audio_engine.amplitude_history)
            result.distortions = np.array(self.audio_engine.distortion_history)
        
        return result


def compare_integrators(
    system: PhysicalSystem,
    position: np.ndarray,
    velocity: np.ndarray,
    config: Optional[SimulationConfig] = None,
    integrators: Optional[List[str]] = None
) -> dict:
    """
    Run the same simulation with multiple integrators for comparison.
    
    Args:
        system: The physical system to simulate
        position: Initial position
        velocity: Initial velocity
        config: Simulation configuration
        integrators: List of integrator names (default: all four)
        
    Returns:
        Dictionary mapping integrator name to SimulationResult
    """
    if config is None:
        config = SimulationConfig(realtime=False, audio_enabled=False)
    
    if integrators is None:
        integrators = ['euler', 'rk4', 'verlet', 'leapfrog']
    
    results = {}
    
    for name in integrators:
        integrator = get_integrator(name)
        sim = Simulator(system, integrator, config)
        results[integrator.name] = sim.run_fast(position, velocity)
    
    return results


def print_comparison(results: dict):
    """Print a comparison table of integrator results."""
    print("\n" + "=" * 80)
    print("INTEGRATOR COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Integrator':<20} {'Symplectic':<12} {'Energy Drift':<15} {'Max Deviation':<15}")
    print("-" * 80)
    
    for name, result in results.items():
        symplectic = "YES" if result.is_symplectic else "NO"
        drift = f"{result.energy_drift:+.6f}"
        max_dev = f"{result.max_energy_deviation:.6f}"
        print(f"{name:<20} {symplectic:<12} {drift:<15} {max_dev:<15}")
    
    print("=" * 80)
    print("\nInterpretation:")
    print("- Energy Drift: Change in total energy over simulation (should be ~0)")
    print("- Max Deviation: Largest instantaneous energy error (should be small)")
    print("- Symplectic integrators (Verlet, Leapfrog) show bounded oscillation")
    print("- Non-symplectic (Euler, RK4) show systematic drift")


if __name__ == "__main__":
    from .physics import HarmonicOscillator
    
    print("Testing Simulation Engine")
    print("=" * 50)
    
    # Create a simple harmonic oscillator
    system = HarmonicOscillator(mass=1.0, spring_constant=1.0)
    
    # Initial conditions
    x0 = np.array([1.0])
    v0 = np.array([0.0])
    
    # Run comparison
    config = SimulationConfig(dt=0.01, duration=10.0, realtime=False)
    results = compare_integrators(system, x0, v0, config)
    
    print_comparison(results)
