#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Tests for the Audible Integrator Project
================================================================================

Project:        Week 1 Project 1: The Audible Integrator
Module:         tests/test_integrators.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 21, 2026
Last Updated:   January 21, 2026

License:        MIT License
================================================================================

Comprehensive test suite for numerical integrators and simulation components.
Run with: pytest tests/test_integrators.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integrators import (
    ForwardEuler, RungeKutta4, VelocityVerlet, Leapfrog,
    get_integrator
)
from src.physics import HarmonicOscillator, DoublePendulum
from src.simulation import Simulator, SimulationConfig, compare_integrators


class TestIntegrators:
    """Test numerical integrator implementations."""
    
    def test_forward_euler_exists(self):
        """Test that Forward Euler can be instantiated."""
        integrator = ForwardEuler()
        assert integrator.name == "Forward Euler"
        assert integrator.is_symplectic == False
    
    def test_rk4_exists(self):
        """Test that RK4 can be instantiated."""
        integrator = RungeKutta4()
        assert integrator.name == "Runge-Kutta 4"
        assert integrator.is_symplectic == False
    
    def test_velocity_verlet_exists(self):
        """Test that Velocity Verlet can be instantiated."""
        integrator = VelocityVerlet()
        assert integrator.name == "Velocity Verlet"
        assert integrator.is_symplectic == True
    
    def test_leapfrog_exists(self):
        """Test that Leapfrog can be instantiated."""
        integrator = Leapfrog()
        assert integrator.name == "Leapfrog"
        assert integrator.is_symplectic == True
    
    def test_get_integrator_factory(self):
        """Test the integrator factory function."""
        euler = get_integrator('euler')
        assert isinstance(euler, ForwardEuler)
        
        rk4 = get_integrator('rk4')
        assert isinstance(rk4, RungeKutta4)
        
        verlet = get_integrator('verlet')
        assert isinstance(verlet, VelocityVerlet)
        
        leapfrog = get_integrator('leapfrog')
        assert isinstance(leapfrog, Leapfrog)
    
    def test_invalid_integrator_raises(self):
        """Test that invalid integrator name raises ValueError."""
        with pytest.raises(ValueError):
            get_integrator('invalid_name')


class TestHarmonicOscillator:
    """Test harmonic oscillator physics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.osc = HarmonicOscillator(mass=1.0, spring_constant=1.0)
    
    def test_natural_frequency(self):
        """Test that natural frequency is correct."""
        # ω = √(k/m) = √(1/1) = 1
        assert np.isclose(self.osc.omega, 1.0)
    
    def test_period(self):
        """Test that period is correct."""
        # T = 2π/ω = 2π
        assert np.isclose(self.osc.period, 2 * np.pi)
    
    def test_energy_conservation_analytical(self):
        """Test that analytical solution conserves energy."""
        x0, v0 = 1.0, 0.0
        E0 = self.osc.total_energy(np.array([x0]), np.array([v0]))
        
        # Check energy at various times
        for t in np.linspace(0, 10, 100):
            x, v = self.osc.analytical_solution(t, x0, v0)
            E = self.osc.total_energy(np.array([x]), np.array([v]))
            assert np.isclose(E, E0, rtol=1e-10)
    
    def test_returns_to_start_after_period(self):
        """Test that oscillator returns to start after one period."""
        x0, v0 = 1.0, 0.0
        x_T, v_T = self.osc.analytical_solution(self.osc.period, x0, v0)
        
        assert np.isclose(x_T, x0, rtol=1e-10)
        assert np.isclose(v_T, v0, atol=1e-10)


class TestSimulation:
    """Test simulation engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.system = HarmonicOscillator(mass=1.0, spring_constant=1.0)
        self.x0 = np.array([1.0])
        self.v0 = np.array([0.0])
        self.config = SimulationConfig(
            dt=0.01,
            duration=10.0,
            realtime=False,
            audio_enabled=False
        )
    
    def test_simulation_runs(self):
        """Test that simulation completes without error."""
        integrator = VelocityVerlet()
        sim = Simulator(self.system, integrator, self.config)
        result = sim.run_fast(self.x0, self.v0)
        
        assert result is not None
        assert len(result.times) > 0
        assert len(result.energies) == len(result.times)
    
    def test_verlet_conserves_energy(self):
        """Test that Velocity Verlet conserves energy."""
        integrator = VelocityVerlet()
        sim = Simulator(self.system, integrator, self.config)
        result = sim.run_fast(self.x0, self.v0)
        
        # Energy drift should be very small
        assert abs(result.energy_drift) < 0.001  # Less than 0.1%
    
    def test_euler_does_not_conserve_energy(self):
        """Test that Forward Euler has significant energy drift."""
        integrator = ForwardEuler()
        sim = Simulator(self.system, integrator, self.config)
        result = sim.run_fast(self.x0, self.v0)
        
        # Forward Euler should have noticeable drift
        assert abs(result.energy_drift) > 0.01  # More than 1%
    
    def test_symplectic_integrators_better_than_euler(self):
        """Test that symplectic integrators outperform Euler."""
        results = compare_integrators(
            self.system, self.x0, self.v0, self.config
        )
        
        euler_drift = abs(results['Forward Euler'].energy_drift)
        verlet_drift = abs(results['Velocity Verlet'].energy_drift)
        leapfrog_drift = abs(results['Leapfrog'].energy_drift)
        
        assert verlet_drift < euler_drift
        assert leapfrog_drift < euler_drift
    
    def test_result_has_expected_fields(self):
        """Test that simulation result contains all expected fields."""
        integrator = VelocityVerlet()
        sim = Simulator(self.system, integrator, self.config)
        result = sim.run_fast(self.x0, self.v0)
        
        assert hasattr(result, 'integrator_name')
        assert hasattr(result, 'is_symplectic')
        assert hasattr(result, 'times')
        assert hasattr(result, 'positions')
        assert hasattr(result, 'velocities')
        assert hasattr(result, 'energies')
        assert hasattr(result, 'initial_energy')
        assert hasattr(result, 'final_energy')
        assert hasattr(result, 'energy_drift')


class TestEnergyMapping:
    """Test energy to audio parameter mapping."""
    
    def test_import_sonification(self):
        """Test that sonification module can be imported."""
        from src.sonification import EnergyToAudio, SonificationConfig
        
        config = SonificationConfig()
        mapper = EnergyToAudio(config, initial_energy=1.0)
        
        assert mapper is not None
    
    def test_energy_to_frequency(self):
        """Test energy to frequency mapping."""
        from src.sonification import EnergyToAudio, SonificationConfig
        
        config = SonificationConfig(base_frequency=220.0)
        mapper = EnergyToAudio(config, initial_energy=1.0)
        
        # At reference energy, should be base frequency
        freq_at_ref = mapper.energy_to_frequency(1.0)
        assert np.isclose(freq_at_ref, 220.0, rtol=0.01)
        
        # Higher energy should give higher frequency
        freq_high = mapper.energy_to_frequency(2.0)
        assert freq_high > freq_at_ref
    
    def test_energy_to_amplitude(self):
        """Test energy to amplitude mapping."""
        from src.sonification import EnergyToAudio, SonificationConfig
        
        config = SonificationConfig()
        mapper = EnergyToAudio(config, initial_energy=1.0)
        
        # Higher energy should give higher amplitude
        amp_low = mapper.energy_to_amplitude(0.5)
        amp_high = mapper.energy_to_amplitude(2.0)
        
        assert amp_high > amp_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
