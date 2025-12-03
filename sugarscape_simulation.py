"""
Sugarscape Simulation Runner
Runs multiple simulation configurations and saves results to CSV files.
"""

import csv
import os
from datetime import datetime
from sugarscape_model import Sugarscape


def run_simulation(config_name: str, num_agents: int, reproduction: bool, 
                   num_steps: int = 500, save_frequency: int = 10):
    """
    Run a single simulation with specified parameters.
    
    Args:
        config_name: Name for this configuration (used in filenames)
        num_agents: Initial number of agents
        reproduction: Whether reproduction is enabled
        num_steps: Number of time steps to simulate
        save_frequency: Save agent data every N steps
    """
    # Header printout to summarize configuration when run from CLI
    print(f"\n{'='*60}")
    print(f"Running simulation: {config_name}")
    print(f"Initial agents: {num_agents}")
    print(f"Reproduction: {reproduction}")
    print(f"Steps: {num_steps}")
    print(f"{'='*60}\n")
    
    # Create Sugarscape environment
    env = Sugarscape(width=50, height=50, initial_agents=num_agents, 
                     reproduction=reproduction)
    
    # Prepare CSV output files in the current working directory. Two files
    # are created per configuration: one for time-series statistics and one
    # for snapshot agent-level data.
    stats_filename = f"{config_name}_statistics.csv"
    agents_filename = f"{config_name}_agents.csv"

    # Write header rows (overwrite any existing files for a fresh run)
    with open(stats_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'time_step', 'population', 'avg_sugar', 'avg_age', 
            'total_born', 'total_died'
        ])
        writer.writeheader()

    with open(agents_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'time_step', 'agent_id', 'x', 'y', 'sugar', 
            'vision', 'metabolism', 'age', 'max_age'
        ])
        writer.writeheader()
    
    # Run simulation
    # Main simulation loop: advance environment, collect and persist data.
    for step in range(num_steps):
        env.step()

        # Retrieve summary statistics after the step
        stats = env.get_statistics()

        # Append statistics to the CSV (time series)
        with open(stats_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'time_step', 'population', 'avg_sugar', 'avg_age',
                'total_born', 'total_died'
            ])
            writer.writerow(stats)

        # Periodically save a snapshot of all alive agents for later analysis
        if step % save_frequency == 0 or step == num_steps - 1:
            agent_data = env.get_agent_data()
            with open(agents_filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'time_step', 'agent_id', 'x', 'y', 'sugar',
                    'vision', 'metabolism', 'age', 'max_age'
                ])
                for agent_dict in agent_data:
                    agent_dict['time_step'] = step
                    writer.writerow(agent_dict)

        # Periodic console progress so user knows simulation is running
        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{num_steps} - Population: {stats['population']} "
                  f"- Avg Sugar: {stats['avg_sugar']:.2f}")

        # Early exit if all agents have died (no population remains)
        if stats['population'] == 0:
            print(f"\nPopulation died out at step {step}")
            break
    
    # Final statistics
    final_stats = env.get_statistics()
    print(f"\n{'='*60}")
    print(f"Simulation Complete: {config_name}")
    print(f"Final Population: {final_stats['population']}")
    print(f"Total Born: {final_stats['total_born']}")
    print(f"Total Died: {final_stats['total_died']}")
    print(f"Average Sugar: {final_stats['avg_sugar']:.2f}")
    print(f"Average Age: {final_stats['avg_age']:.2f}")
    print(f"{'='*60}\n")
    
    return env, final_stats


def main():
    """Run all simulation configurations."""
    print("="*60)
    print("SUGARSCAPE SIMULATION PROJECT")
    print("Collective and Social Intelligence in Agent-Based Models")
    print("="*60)
    
    # Create output directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"simulation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    # Configuration 1: Base simulation with 100 agents (no reproduction)
    print("\n\n### CONFIGURATION 1: Base Simulation ###")
    env1, stats1 = run_simulation(
        config_name="config1_base",
        num_agents=100,
        reproduction=False,
        num_steps=500,
        save_frequency=10
    )
    
    # Configuration 2: Simulation with reproduction enabled
    print("\n\n### CONFIGURATION 2: With Reproduction ###")
    env2, stats2 = run_simulation(
        config_name="config2_reproduction",
        num_agents=100,
        reproduction=True,
        num_steps=500,
        save_frequency=10
    )
    
    # Configuration 3: Smaller initial population with reproduction
    print("\n\n### CONFIGURATION 3: Small Population with Reproduction ###")
    env3, stats3 = run_simulation(
        config_name="config3_small_population",
        num_agents=50,
        reproduction=True,
        num_steps=500,
        save_frequency=10
    )
    
    # Summary report
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"\nConfiguration 1 (Base):")
    print(f"  Final Population: {stats1['population']}")
    print(f"  Total Born: {stats1['total_born']}")
    print(f"  Total Died: {stats1['total_died']}")
    
    print(f"\nConfiguration 2 (Reproduction):")
    print(f"  Final Population: {stats2['population']}")
    print(f"  Total Born: {stats2['total_born']}")
    print(f"  Total Died: {stats2['total_died']}")
    
    print(f"\nConfiguration 3 (Small + Reproduction):")
    print(f"  Final Population: {stats3['population']}")
    print(f"  Total Born: {stats3['total_born']}")
    print(f"  Total Died: {stats3['total_died']}")
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
