#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Main Entry Point - Command Line Interface
================================================================================

Project:        Week 1 Project 1: The Audible Integrator
Module:         main.py (CLI Entry Point)

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 21, 2026
Last Updated:   January 21, 2026

License:        MIT License
================================================================================

Main entry point for running simulations from the command line.

Usage:
    python main.py                    # Run comparison with defaults
    python main.py --integrator euler # Run single integrator
    python main.py --audio            # Run with audio sonification
    python main.py --save-plots       # Save comparison plots
"""

import argparse
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.integrators import get_integrator, INTEGRATOR_COMPARISON
from src.physics import HarmonicOscillator, DoublePendulum
from src.simulation import (
    Simulator, 
    SimulationConfig, 
    compare_integrators,
    print_comparison
)
from src.visualization import (
    plot_energy_comparison,
    plot_phase_space,
    create_summary_figure
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="The Audible Integrator - Sonification of Energy Drift",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Compare all integrators
  python main.py --integrator euler       # Run Forward Euler only
  python main.py --dt 0.05 --duration 20  # Custom parameters
  python main.py --system pendulum        # Double pendulum
  python main.py --save-plots             # Save figures to disk
  python main.py --audio                  # Enable audio sonification
        """
    )
    
    parser.add_argument(
        '--integrator', '-i',
        type=str,
        choices=['euler', 'rk4', 'verlet', 'leapfrog', 'all'],
        default='all',
        help='Which integrator to use (default: all for comparison)'
    )
    
    parser.add_argument(
        '--system', '-s',
        type=str,
        choices=['oscillator', 'pendulum'],
        default='oscillator',
        help='Physical system to simulate'
    )
    
    parser.add_argument(
        '--dt',
        type=float,
        default=0.02,
        help='Time step in seconds (default: 0.02)'
    )
    
    parser.add_argument(
        '--duration', '-t',
        type=float,
        default=10.0,
        help='Simulation duration in seconds (default: 10.0)'
    )
    
    parser.add_argument(
        '--audio', '-a',
        action='store_true',
        help='Enable real-time audio sonification'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save plots to disk'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='output',
        help='Directory for output files (default: output)'
    )
    
    parser.add_argument(
        '--show-info',
        action='store_true',
        help='Show integrator comparison table'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Show info if requested
    if args.show_info:
        print(INTEGRATOR_COMPARISON)
        return
    
    print("=" * 70)
    print("THE AUDIBLE INTEGRATOR")
    print("Sonification of Energy Drift in Numerical Integration")
    print("=" * 70)
    
    # Create physical system
    if args.system == 'oscillator':
        system = HarmonicOscillator(mass=1.0, spring_constant=1.0)
        initial_position = np.array([1.0])  # Start at x = 1
        initial_velocity = np.array([0.0])  # Start at rest
        print(f"\nSystem: Simple Harmonic Oscillator (m=1, k=1)")
        print(f"Period T = {system.period:.4f} s")
    else:
        system = DoublePendulum(m1=1.0, m2=1.0, L1=1.0, L2=1.0)
        initial_position = np.array([np.pi/2, np.pi/2])  # 90 degrees
        initial_velocity = np.array([0.0, 0.0])
        print(f"\nSystem: Double Pendulum (m₁=m₂=1, L₁=L₂=1)")
    
    print(f"Initial position: {initial_position}")
    print(f"Initial velocity: {initial_velocity}")
    print(f"Time step: {args.dt} s")
    print(f"Duration: {args.duration} s")
    
    # Create output directory if saving plots
    if args.save_plots:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Run simulation(s)
    if args.integrator == 'all':
        # Compare all integrators
        print("\n" + "-" * 70)
        print("Running comparison of all four integrators...")
        print("-" * 70)
        
        config = SimulationConfig(
            dt=args.dt,
            duration=args.duration,
            realtime=False,
            audio_enabled=False
        )
        
        results = compare_integrators(system, initial_position, initial_velocity, config)
        print_comparison(results)
        
        # Generate plots
        print("\nGenerating plots...")
        
        try:
            import matplotlib
            matplotlib.use('Agg' if args.save_plots else 'TkAgg')
            import matplotlib.pyplot as plt
            
            fig1 = plot_energy_comparison(
                results,
                title=f"Energy Conservation: {args.system.title()}",
                save_path=str(Path(args.output_dir) / "energy_comparison.png") if args.save_plots else None
            )
            
            fig2 = plot_phase_space(
                results,
                title=f"Phase Space: {args.system.title()}",
                save_path=str(Path(args.output_dir) / "phase_space.png") if args.save_plots else None
            )
            
            fig3 = create_summary_figure(
                results,
                title=f"The Audible Integrator: {args.system.title()}",
                save_path=str(Path(args.output_dir) / "summary.png") if args.save_plots else None
            )
            
            if args.save_plots:
                print(f"Plots saved to {args.output_dir}/")
            else:
                plt.show()
                
        except ImportError as e:
            print(f"Could not generate plots: {e}")
            print("Install matplotlib to enable plotting: pip install matplotlib")
    
    else:
        # Single integrator
        print(f"\n" + "-" * 70)
        print(f"Running {args.integrator.upper()} integrator...")
        print("-" * 70)
        
        integrator = get_integrator(args.integrator)
        config = SimulationConfig(
            dt=args.dt,
            duration=args.duration,
            realtime=args.audio,
            audio_enabled=args.audio
        )
        
        sim = Simulator(system, integrator, config)
        
        if args.audio:
            print("\n🔊 Audio sonification enabled!")
            print("Listen to the energy drift...")
            print("(Press Ctrl+C to stop)\n")
            
            try:
                result = sim.run(initial_position, initial_velocity)
            except KeyboardInterrupt:
                print("\nSimulation stopped by user.")
                return
        else:
            result = sim.run_fast(initial_position, initial_velocity)
        
        # Print results
        print(f"\nResults for {result.integrator_name}:")
        print(f"  Symplectic: {'Yes' if result.is_symplectic else 'No'}")
        print(f"  Initial Energy: {result.initial_energy:.6f} J")
        print(f"  Final Energy: {result.final_energy:.6f} J")
        print(f"  Energy Drift: {result.energy_drift*100:+.6f}%")
        print(f"  Max Deviation: {result.max_energy_deviation*100:.6f}%")
        
        if result.is_symplectic:
            print("\n✅ Symplectic integrator - Energy bounded!")
        elif abs(result.energy_drift) > 0.1:
            print("\n⚠️  WARNING: Significant energy drift detected!")
            print("   This would cause 'explosions' in long MD simulations.")
    
    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)
    
    # Hint about Streamlit app
    print("\nTip: Run the interactive web app with:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
