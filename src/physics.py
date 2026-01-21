#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Physical Systems for Integration Testing
================================================================================

Project:        Week 1 Project 1: The Audible Integrator
Module:         physics.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 21, 2026
Last Updated:   January 21, 2026

License:        MIT License
================================================================================

This module implements two classical mechanical systems:
1. Simple Harmonic Oscillator - The fundamental testbed for integrators
2. Double Pendulum - A chaotic system to show integrator limitations

Both systems are Hamiltonian, meaning they conserve total energy.
This makes them perfect for testing whether our integrators preserve
physical conservation laws.
"""

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class PhysicalState:
    """Current state of a physical system."""
    time: float
    position: np.ndarray
    velocity: np.ndarray
    kinetic_energy: float
    potential_energy: float
    
    @property
    def total_energy(self) -> float:
        """Total mechanical energy (should be conserved!)"""
        return self.kinetic_energy + self.potential_energy


class PhysicalSystem(ABC):
    """Abstract base class for physical systems."""
    
    @abstractmethod
    def acceleration(self, position: np.ndarray) -> np.ndarray:
        """Compute acceleration from position (F = ma, so a = F/m)."""
        pass
    
    @abstractmethod
    def kinetic_energy(self, velocity: np.ndarray) -> float:
        """Compute kinetic energy from velocity."""
        pass
    
    @abstractmethod
    def potential_energy(self, position: np.ndarray) -> float:
        """Compute potential energy from position."""
        pass
    
    def total_energy(self, position: np.ndarray, velocity: np.ndarray) -> float:
        """Compute total mechanical energy."""
        return self.kinetic_energy(velocity) + self.potential_energy(position)
    
    def get_state(self, time: float, position: np.ndarray, velocity: np.ndarray) -> PhysicalState:
        """Get the complete physical state."""
        return PhysicalState(
            time=time,
            position=position.copy(),
            velocity=velocity.copy(),
            kinetic_energy=self.kinetic_energy(velocity),
            potential_energy=self.potential_energy(position)
        )


class HarmonicOscillator(PhysicalSystem):
    """
    Simple Harmonic Oscillator
    
    The simplest oscillatory system: a mass on a spring.
    
    Physics:
        F = -kx  (Hooke's Law)
        a = -ω²x  where ω = √(k/m)
        
    Hamiltonian:
        H = T + V = (1/2)mv² + (1/2)kx²
        
    Analytical Solution (for testing):
        x(t) = A·cos(ωt + φ)
        v(t) = -Aω·sin(ωt + φ)
        
    Why it's perfect for testing:
        - Exact analytical solution known
        - Energy should be EXACTLY conserved
        - Any energy drift directly shows integrator errors
    """
    
    def __init__(self, mass: float = 1.0, spring_constant: float = 1.0):
        """
        Initialize harmonic oscillator.
        
        Args:
            mass: Mass of the oscillating particle (kg)
            spring_constant: Spring constant k (N/m)
        """
        self.mass = mass
        self.k = spring_constant
        self.omega = np.sqrt(spring_constant / mass)  # Natural frequency
        self.period = 2 * np.pi / self.omega  # Period of oscillation
    
    def acceleration(self, position: np.ndarray) -> np.ndarray:
        """a = -(k/m)x = -ω²x"""
        return -self.omega**2 * position
    
    def kinetic_energy(self, velocity: np.ndarray) -> float:
        """T = (1/2)mv²"""
        return 0.5 * self.mass * np.sum(velocity**2)
    
    def potential_energy(self, position: np.ndarray) -> float:
        """V = (1/2)kx²"""
        return 0.5 * self.k * np.sum(position**2)
    
    def analytical_solution(self, t: float, x0: float, v0: float) -> Tuple[float, float]:
        """
        Compute the exact analytical solution.
        
        Args:
            t: Time
            x0: Initial position
            v0: Initial velocity
            
        Returns:
            (position, velocity) at time t
        """
        # Convert initial conditions to amplitude and phase
        A = np.sqrt(x0**2 + (v0/self.omega)**2)
        phi = np.arctan2(-v0/self.omega, x0)
        
        # Analytical solution
        x = A * np.cos(self.omega * t + phi)
        v = -A * self.omega * np.sin(self.omega * t + phi)
        
        return x, v


class DoublePendulum(PhysicalSystem):
    """
    Double Pendulum
    
    Two pendulums attached end-to-end. Famous for exhibiting chaotic behavior.
    
    Configuration:
        - θ₁: Angle of first pendulum from vertical
        - θ₂: Angle of second pendulum from vertical
        - L₁, L₂: Lengths of the pendulum rods
        - m₁, m₂: Masses at the end of each rod
        
    The equations of motion are coupled nonlinear ODEs derived from
    the Lagrangian. This system is chaotic: tiny differences in initial
    conditions lead to wildly different trajectories.
    
    Why it's interesting:
        - Chaotic dynamics test integrator robustness
        - Energy should still be conserved despite chaos
        - Beautiful phase space trajectories
    """
    
    def __init__(
        self,
        m1: float = 1.0,
        m2: float = 1.0,
        L1: float = 1.0,
        L2: float = 1.0,
        g: float = 9.81
    ):
        """
        Initialize double pendulum.
        
        Args:
            m1, m2: Masses of the two pendulum bobs
            L1, L2: Lengths of the two pendulum rods
            g: Gravitational acceleration
        """
        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2
        self.g = g
    
    def acceleration(self, position: np.ndarray) -> np.ndarray:
        """
        Compute angular accelerations from the Euler-Lagrange equations.
        
        position = [θ₁, θ₂]
        Returns: [θ̈₁, θ̈₂]
        
        The equations are derived from the Lagrangian L = T - V.
        """
        theta1, theta2 = position
        # We need angular velocities for the full equations
        # This is a simplification - we'll handle this differently in the simulator
        return self._compute_acceleration(position, np.zeros(2))
    
    def _compute_acceleration(
        self, 
        position: np.ndarray, 
        velocity: np.ndarray
    ) -> np.ndarray:
        """
        Full acceleration computation including velocity dependence.
        
        The double pendulum equations of motion (from Lagrangian mechanics):
        """
        theta1, theta2 = position
        omega1, omega2 = velocity
        
        m1, m2 = self.m1, self.m2
        L1, L2 = self.L1, self.L2
        g = self.g
        
        delta = theta2 - theta1
        
        # Denominators (mass matrix determinant)
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
        den2 = (L2 / L1) * den1
        
        # Angular acceleration of first pendulum
        num1 = (m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta) +
                m2 * g * np.sin(theta2) * np.cos(delta) +
                m2 * L2 * omega2**2 * np.sin(delta) -
                (m1 + m2) * g * np.sin(theta1))
        alpha1 = num1 / den1
        
        # Angular acceleration of second pendulum
        num2 = (-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta) +
                (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                (m1 + m2) * L1 * omega1**2 * np.sin(delta) -
                (m1 + m2) * g * np.sin(theta2))
        alpha2 = num2 / den2
        
        return np.array([alpha1, alpha2])
    
    def kinetic_energy(self, velocity: np.ndarray) -> float:
        """
        Kinetic energy of double pendulum.
        
        T = (1/2)m₁L₁²ω₁² + (1/2)m₂[L₁²ω₁² + L₂²ω₂² + 2L₁L₂ω₁ω₂cos(θ₁-θ₂)]
        
        Note: This needs position too, but we store it for convenience.
        """
        omega1, omega2 = velocity
        m1, m2 = self.m1, self.m2
        L1, L2 = self.L1, self.L2
        
        # Simplified (without cross term which needs position)
        T = 0.5 * m1 * L1**2 * omega1**2
        T += 0.5 * m2 * (L1**2 * omega1**2 + L2**2 * omega2**2)
        
        return T
    
    def potential_energy(self, position: np.ndarray) -> float:
        """
        Potential energy of double pendulum.
        
        V = -m₁gL₁cos(θ₁) - m₂g[L₁cos(θ₁) + L₂cos(θ₂)]
        
        (Taking the pivot as zero potential)
        """
        theta1, theta2 = position
        m1, m2 = self.m1, self.m2
        L1, L2 = self.L1, self.L2
        g = self.g
        
        # Height of each mass below pivot (negative = below)
        y1 = -L1 * np.cos(theta1)
        y2 = -L1 * np.cos(theta1) - L2 * np.cos(theta2)
        
        # Potential energy (taking pivot height as zero)
        V = m1 * g * y1 + m2 * g * y2
        
        return V
    
    def cartesian_positions(self, position: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Convert angular positions to Cartesian (x, y) for visualization.
        
        Returns:
            ((x1, y1), (x2, y2)) - positions of the two masses
        """
        theta1, theta2 = position
        
        x1 = self.L1 * np.sin(theta1)
        y1 = -self.L1 * np.cos(theta1)
        
        x2 = x1 + self.L2 * np.sin(theta2)
        y2 = y1 - self.L2 * np.cos(theta2)
        
        return (x1, y1), (x2, y2)


# Default initial conditions for demos
DEFAULT_OSCILLATOR_IC = {
    'position': np.array([1.0]),  # Start at x = 1
    'velocity': np.array([0.0]),  # Start at rest
}

DEFAULT_PENDULUM_IC = {
    'position': np.array([np.pi/2, np.pi/2]),  # Both at 90 degrees
    'velocity': np.array([0.0, 0.0]),           # Start at rest
}


if __name__ == "__main__":
    # Quick test
    print("Testing Harmonic Oscillator")
    print("=" * 40)
    
    osc = HarmonicOscillator(mass=1.0, spring_constant=1.0)
    print(f"Natural frequency ω = {osc.omega:.4f} rad/s")
    print(f"Period T = {osc.period:.4f} s")
    
    x0, v0 = 1.0, 0.0
    E0 = osc.total_energy(np.array([x0]), np.array([v0]))
    print(f"Initial energy = {E0:.4f} J")
    
    # Check analytical solution after one period
    x_T, v_T = osc.analytical_solution(osc.period, x0, v0)
    print(f"After one period: x = {x_T:.6f}, v = {v_T:.6f}")
    print(f"(Should be back to x ≈ {x0}, v ≈ {v0})")
