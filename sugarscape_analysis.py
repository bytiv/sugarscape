"""
Sugarscape Analysis and Visualization
Analyzes simulation results and creates visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


def load_simulation_data(config_name: str):
    """
    Load time-series statistics and agent snapshot CSVs for a given
    configuration name.

    Returns a tuple `(stats_df, agents_df)` where `agents_df` may be
    `None` if no agent-level CSV was produced for the configuration.
    """
    stats_file = f"{config_name}_statistics.csv"
    agents_file = f"{config_name}_agents.csv"

    if not os.path.exists(stats_file):
        print(f"Warning: {stats_file} not found")
        return None, None

    stats_df = pd.read_csv(stats_file)

    # Agent data is optional; some runs may not have produced snapshots
    if os.path.exists(agents_file):
        agents_df = pd.read_csv(agents_file)
    else:
        agents_df = None

    return stats_df, agents_df


def plot_population_dynamics(configs: dict):
    """Plot population over time for all configurations.

    `configs` should be a mapping from filename prefix to human label,
    e.g. `{'config1_base': 'Base (100 agents)'}`.
    """
    plt.figure(figsize=(12, 6))
    
    for config_name, label in configs.items():
        stats_df, _ = load_simulation_data(config_name)
        if stats_df is not None:
            plt.plot(stats_df['time_step'], stats_df['population'], 
                    label=label, linewidth=2)
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Population', fontsize=12)
    plt.title('Population Dynamics Across Configurations', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('population_dynamics.png', dpi=300)
    print("Saved: population_dynamics.png")
    plt.close()


def plot_wealth_distribution(configs: dict):
    """Plot average sugar (mean wealth) over time for each configuration."""
    plt.figure(figsize=(12, 6))
    
    for config_name, label in configs.items():
        stats_df, _ = load_simulation_data(config_name)
        if stats_df is not None:
            plt.plot(stats_df['time_step'], stats_df['avg_sugar'], 
                    label=label, linewidth=2)
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Average Sugar (Wealth)', fontsize=12)
    plt.title('Average Wealth Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('wealth_distribution.png', dpi=300)
    print("Saved: wealth_distribution.png")
    plt.close()


def plot_age_distribution(configs: dict):
    """Plot average agent age over time for each configuration."""
    plt.figure(figsize=(12, 6))
    
    for config_name, label in configs.items():
        stats_df, _ = load_simulation_data(config_name)
        if stats_df is not None:
            plt.plot(stats_df['time_step'], stats_df['avg_age'], 
                    label=label, linewidth=2)
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Average Age', fontsize=12)
    plt.title('Average Age Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('age_distribution.png', dpi=300)
    print("Saved: age_distribution.png")
    plt.close()


def plot_demographics(config_name: str, label: str):
    """
    Create a 2x2 figure of demographic time series for a single
    configuration: population, average wealth, average age, and births/deaths.
    """
    stats_df, _ = load_simulation_data(config_name)
    if stats_df is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Population
    axes[0, 0].plot(stats_df['time_step'], stats_df['population'], 
                    color='blue', linewidth=2)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Population')
    axes[0, 0].set_title('Population Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Average Sugar
    axes[0, 1].plot(stats_df['time_step'], stats_df['avg_sugar'], 
                    color='green', linewidth=2)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Average Sugar')
    axes[0, 1].set_title('Average Wealth Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Average Age
    axes[1, 0].plot(stats_df['time_step'], stats_df['avg_age'], 
                    color='red', linewidth=2)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Average Age')
    axes[1, 0].set_title('Average Age Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Births and Deaths
    axes[1, 1].plot(stats_df['time_step'], stats_df['total_born'], 
                    label='Total Born', color='blue', linewidth=2)
    axes[1, 1].plot(stats_df['time_step'], stats_df['total_died'], 
                    label='Total Died', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Births and Deaths')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Demographics: {label}', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    filename = f'{config_name}_demographics.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def analyze_wealth_inequality(config_name: str):
    """
    Compute simple wealth inequality statistics from the final agent
    snapshot for `config_name`.

    Returns a dictionary containing a Gini coefficient and basic
    descriptive statistics (mean, median, std, percentiles). If agent
    data is missing or empty, returns `None`.
    """
    _, agents_df = load_simulation_data(config_name)
    if agents_df is None:
        return None

    # Select the final snapshot in time (agents were saved periodically)
    final_step = agents_df['time_step'].max()
    final_agents = agents_df[agents_df['time_step'] == final_step]

    if len(final_agents) == 0:
        return None

    # Calculate Gini coefficient in a simple O(n) formula. Gini is a
    # measure of inequality where 0 means perfect equality and 1 means
    # maximal inequality. The implementation below assumes non-negative
    # sugar (wealth) values.
    sugar_values = sorted(final_agents['sugar'].values)
    n = len(sugar_values)

    if n == 0 or sum(sugar_values) == 0:
        gini = 0
    else:
        # The expression below computes the Gini using the mean of pairwise
        # differences equivalent; we use the rank formulation for efficiency.
        gini = (2 * sum((i + 1) * sugar_values[i] for i in range(n))) / (n * sum(sugar_values)) - (n + 1) / n

    # Package additional descriptive statistics for reporting
    wealth_stats = {
        'gini': gini,
        'mean': final_agents['sugar'].mean(),
        'median': final_agents['sugar'].median(),
        'std': final_agents['sugar'].std(),
        'min': final_agents['sugar'].min(),
        'max': final_agents['sugar'].max(),
        'p10': final_agents['sugar'].quantile(0.1),
        'p90': final_agents['sugar'].quantile(0.9),
    }

    return wealth_stats


def create_summary_report(configs: dict):
    """Create and print a compact summary report for each configuration.

    The report shows final population, cumulative births/deaths, final
    average wealth/age, and a small wealth-inequality summary when agent
    snapshots are available.
    """
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY REPORT")
    print("="*80 + "\n")
    
    for config_name, label in configs.items():
        print(f"\n{label}")
        print("-" * 60)
        
        stats_df, agents_df = load_simulation_data(config_name)
        
        if stats_df is not None:
            final_stats = stats_df.iloc[-1]
            print(f"  Final Population: {int(final_stats['population'])}")
            print(f"  Total Born: {int(final_stats['total_born'])}")
            print(f"  Total Died: {int(final_stats['total_died'])}")
            print(f"  Final Avg Sugar: {final_stats['avg_sugar']:.2f}")
            print(f"  Final Avg Age: {final_stats['avg_age']:.2f}")
            
            # Population statistics
            max_pop = stats_df['population'].max()
            min_pop = stats_df['population'].min()
            print(f"  Max Population: {int(max_pop)}")
            print(f"  Min Population: {int(min_pop)}")
            
            # Wealth inequality
            wealth_stats = analyze_wealth_inequality(config_name)
            if wealth_stats:
                print(f"\n  Wealth Inequality Metrics:")
                print(f"    Gini Coefficient: {wealth_stats['gini']:.3f}")
                print(f"    Mean Sugar: {wealth_stats['mean']:.2f}")
                print(f"    Median Sugar: {wealth_stats['median']:.2f}")
                print(f"    Std Dev: {wealth_stats['std']:.2f}")
                print(f"    10th Percentile: {wealth_stats['p10']:.2f}")
                print(f"    90th Percentile: {wealth_stats['p90']:.2f}")


def main():
    """Main analysis routine that finds the latest simulation results
    and produces plots and textual summaries.
    """
    print("\n" + "="*80)
    print("SUGARSCAPE SIMULATION ANALYSIS")
    print("="*80 + "\n")
    
    # Find the most recent simulation results directory
    result_dirs = sorted(glob.glob("simulation_results_*"), reverse=True)
    
    if result_dirs:
        result_dir = result_dirs[0]
        print(f"Analyzing results from: {result_dir}\n")
        os.chdir(result_dir)
    else:
        print("No simulation results found. Run simulation first.")
        return
    
    # Define configurations to analyze
    configs = {
        'config1_base': 'Base (100 agents, no reproduction)',
        'config2_reproduction': 'Reproduction (100 agents)',
        'config3_small_population': 'Small Population (50 agents + reproduction)',
    }
    
    # Create visualizations
    print("Creating visualizations...")
    plot_population_dynamics(configs)
    plot_wealth_distribution(configs)
    plot_age_distribution(configs)
    
    # Create detailed demographics for each configuration
    for config_name, label in configs.items():
        plot_demographics(config_name, label)
    
    # Create summary report
    create_summary_report(configs)
    
    print("\n" + "="*80)
    print("Analysis complete! Check the generated PNG files.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
