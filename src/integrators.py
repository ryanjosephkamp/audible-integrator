#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Numerical Integration Methods for Classical Mechanics
================================================================================

Project:        Week 1 Project 1: The Audible Integrator
Module:         integrators.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 21, 2026
Last Updated:   January 21, 2026

License:        MIT License
================================================================================

This module implements four numerical integrators from scratch:
1. Forward Euler - First-order, non-symplectic (UNSTABLE for oscillatory systems)
2. Runge-Kutta 4 (RK4) - Fourth-order, non-symplectic (accurate but energy drifts)
3. Velocity Verlet - Second-order, symplectic (STABLE for molecular dynamics)
4. Leapfrog - Second-order, symplectic (equivalent to Velocity Verlet)

Key Concept: Symplectic integrators preserve the phase space volume (Liouville's theorem),
which is essential for long-time stability in Hamiltonian systems.
"""

import numpy as np
from typing import Callable, Tuple
from dataclasses import dataclass


@dataclass
class IntegratorResult:
    """Container for integration results at each timestep."""
    time: float
    position: np.ndarray
    velocity: np.ndarray
    energy: float


class NumericalIntegrator:
    """Base class for numerical integrators."""
    
    def __init__(self, name: str, is_symplectic: bool):
        self.name = name
        self.is_symplectic = is_symplectic
    
    def step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration_func: Callable[[np.ndarray], np.ndarray],
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one integration step.
        
        Args:
            position: Current position(s)
            velocity: Current velocity(ies)
            acceleration_func: Function that computes acceleration from position
            dt: Time step
            
        Returns:
            Tuple of (new_position, new_velocity)
        """
        raise NotImplementedError


class ForwardEuler(NumericalIntegrator):
    """
    Forward Euler Method (First-Order)
    
    The simplest integration method. Uses current derivatives to estimate next state.
    
    Update equations:
        x_{n+1} = x_n + v_n * dt
        v_{n+1} = v_n + a_n * dt
    
    Properties:
        - Order: 1 (local error ~ O(dt²), global error ~ O(dt))
        - Symplectic: NO
        - Energy behavior: GROWS exponentially for oscillatory systems
        
    This is the "villain" of our demo - it shows what goes wrong!
    """
    
    def __init__(self):
        super().__init__("Forward Euler", is_symplectic=False)
    
    def step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration_func: Callable[[np.ndarray], np.ndarray],
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Compute acceleration at current position
        acceleration = acceleration_func(position)
        
        # Simple forward step
        new_position = position + velocity * dt
        new_velocity = velocity + acceleration * dt
        
        return new_position, new_velocity


class RungeKutta4(NumericalIntegrator):
    """
    Runge-Kutta 4th Order Method (RK4)
    
    The "gold standard" for general ODE integration. Uses weighted average
    of four derivative evaluations per step.
    
    For a system dy/dt = f(t, y):
        k1 = f(t_n, y_n)
        k2 = f(t_n + dt/2, y_n + dt*k1/2)
        k3 = f(t_n + dt/2, y_n + dt*k2/2)
        k4 = f(t_n + dt, y_n + dt*k3)
        y_{n+1} = y_n + (dt/6)(k1 + 2*k2 + 2*k3 + k4)
    
    Properties:
        - Order: 4 (local error ~ O(dt⁵), global error ~ O(dt⁴))
        - Symplectic: NO
        - Energy behavior: Drifts slowly (much better than Euler, but not conserved)
        
    Great for general ODEs, but NOT ideal for long-time Hamiltonian dynamics.
    """
    
    def __init__(self):
        super().__init__("Runge-Kutta 4", is_symplectic=False)
    
    def step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration_func: Callable[[np.ndarray], np.ndarray],
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        # We treat this as a system: d/dt [x, v] = [v, a(x)]
        
        # k1: derivatives at current state
        k1_x = velocity
        k1_v = acceleration_func(position)
        
        # k2: derivatives at midpoint using k1
        k2_x = velocity + 0.5 * dt * k1_v
        k2_v = acceleration_func(position + 0.5 * dt * k1_x)
        
        # k3: derivatives at midpoint using k2
        k3_x = velocity + 0.5 * dt * k2_v
        k3_v = acceleration_func(position + 0.5 * dt * k2_x)
        
        # k4: derivatives at endpoint using k3
        k4_x = velocity + dt * k3_v
        k4_v = acceleration_func(position + dt * k3_x)
        
        # Weighted average
        new_position = position + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        new_velocity = velocity + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
        return new_position, new_velocity


class VelocityVerlet(NumericalIntegrator):
    """
    Velocity Verlet Method (Second-Order, Symplectic)
    
    THE standard algorithm for molecular dynamics simulations.
    Named after Loup Verlet who used it for MD in 1967.
    
    Update equations:
        x_{n+1} = x_n + v_n * dt + 0.5 * a_n * dt²
        a_{n+1} = acceleration(x_{n+1})
        v_{n+1} = v_n + 0.5 * (a_n + a_{n+1}) * dt
    
    Properties:
        - Order: 2 (local error ~ O(dt³), global error ~ O(dt²))
        - Symplectic: YES (preserves phase space volume)
        - Energy behavior: Bounded oscillation around true value (NO DRIFT!)
        - Time-reversible: YES
        
    Why it works for MD:
        - Energy conservation (within bounded oscillations)
        - Stable for arbitrarily long simulations
        - Simple and efficient (one force evaluation per step)
    """
    
    def __init__(self):
        super().__init__("Velocity Verlet", is_symplectic=True)
        self._prev_acceleration = None
    
    def step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration_func: Callable[[np.ndarray], np.ndarray],
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Get current acceleration
        if self._prev_acceleration is None:
            acceleration = acceleration_func(position)
        else:
            acceleration = self._prev_acceleration
        
        # Update position (full step)
        new_position = position + velocity * dt + 0.5 * acceleration * dt**2
        
        # Calculate new acceleration
        new_acceleration = acceleration_func(new_position)
        
        # Update velocity using average of old and new acceleration
        new_velocity = velocity + 0.5 * (acceleration + new_acceleration) * dt
        
        # Store for next step
        self._prev_acceleration = new_acceleration
        
        return new_position, new_velocity
    
    def reset(self):
        """Reset stored acceleration for new simulation."""
        self._prev_acceleration = None


class Leapfrog(NumericalIntegrator):
    """
    Leapfrog Method (Second-Order, Symplectic)
    
    Position and velocity are evaluated at interleaved ("staggered") time points.
    Mathematically equivalent to Velocity Verlet but organized differently.
    
    Update equations (kick-drift-kick form):
        v_{n+1/2} = v_n + a_n * dt/2          # Half-step velocity (kick)
        x_{n+1} = x_n + v_{n+1/2} * dt        # Full-step position (drift)
        a_{n+1} = acceleration(x_{n+1})        # New acceleration
        v_{n+1} = v_{n+1/2} + a_{n+1} * dt/2  # Complete velocity (kick)
    
    Properties:
        - Order: 2
        - Symplectic: YES
        - Energy behavior: Bounded oscillation (same as Velocity Verlet)
        - Time-reversible: YES
        
    The name comes from how position and velocity "leap" over each other
    in the integration sequence.
    """
    
    def __init__(self):
        super().__init__("Leapfrog", is_symplectic=True)
    
    def step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration_func: Callable[[np.ndarray], np.ndarray],
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Kick: half-step velocity update
        acceleration = acceleration_func(position)
        velocity_half = velocity + 0.5 * acceleration * dt
        
        # Drift: full-step position update
        new_position = position + velocity_half * dt
        
        # Kick: complete velocity update with new acceleration
        new_acceleration = acceleration_func(new_position)
        new_velocity = velocity_half + 0.5 * new_acceleration * dt
        
        return new_position, new_velocity


# Factory function for easy integrator creation
def get_integrator(name: str) -> NumericalIntegrator:
    """
    Factory function to get an integrator by name.
    
    Args:
        name: One of 'euler', 'rk4', 'verlet', 'leapfrog'
        
    Returns:
        NumericalIntegrator instance
    """
    integrators = {
        'euler': ForwardEuler,
        'forward_euler': ForwardEuler,
        'rk4': RungeKutta4,
        'runge_kutta': RungeKutta4,
        'verlet': VelocityVerlet,
        'velocity_verlet': VelocityVerlet,
        'leapfrog': Leapfrog,
    }
    
    name_lower = name.lower().replace(' ', '_').replace('-', '_')
    if name_lower not in integrators:
        available = list(integrators.keys())
        raise ValueError(f"Unknown integrator '{name}'. Available: {available}")
    
    return integrators[name_lower]()


# Comparison table for educational purposes
INTEGRATOR_COMPARISON = """
╔═══════════════════╦═══════════╦════════════╦═══════════════════════════════════╗
║    Integrator     ║   Order   ║ Symplectic ║        Energy Behavior            ║
╠═══════════════════╬═══════════╬════════════╬═══════════════════════════════════╣
║ Forward Euler     ║    1      ║     NO     ║ Exponential growth (UNSTABLE)     ║
║ Runge-Kutta 4     ║    4      ║     NO     ║ Slow drift (not conserved)        ║
║ Velocity Verlet   ║    2      ║    YES     ║ Bounded oscillation (STABLE)      ║
║ Leapfrog          ║    2      ║    YES     ║ Bounded oscillation (STABLE)      ║
╚═══════════════════╩═══════════╩════════════╩═══════════════════════════════════╝

Key Insight: For molecular dynamics and Hamiltonian systems, symplectic integrators
(Verlet, Leapfrog) are ESSENTIAL. They may be lower order than RK4, but they 
preserve the geometric structure of phase space, ensuring long-term stability.
"""


if __name__ == "__main__":
    print("Numerical Integrators for Classical Mechanics")
    print("=" * 50)
    print(INTEGRATOR_COMPARISON)
