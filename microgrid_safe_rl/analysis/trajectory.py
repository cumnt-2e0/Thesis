#!/usr/bin/env python3
"""
Analyze trajectory data to understand agent behavior.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Load trajectory data
print("Loading trajectory data...")
df = pd.read_csv("results/trajectory_with_fault.csv")

print(f"Loaded {len(df)} steps across {df['episode'].nunique()} episodes")
print(f"\nColumns: {list(df.columns)}")

# ============================================================================
# Question 1: What actions is the agent taking?
# ============================================================================
print("\n" + "="*80)
print("ACTION ANALYSIS")
print("="*80)

action_counts = df['action_type'].value_counts()
print("\nAction Distribution (all episodes):")
for action, count in action_counts.items():
    pct = count / len(df) * 100
    print(f"  {action:20s}: {count:5d} ({pct:.1f}%)")

# Count switch operations specifically
switch_actions = df[df['action_type'].str.contains('switch', na=False)]
print(f"\nTotal switch operations: {len(switch_actions)}")
if len(switch_actions) > 0:
    print(f"  Opens:  {len(switch_actions[switch_actions['action_type'] == 'switch_open'])}")
    print(f"  Closes: {len(switch_actions[switch_actions['action_type'] == 'switch_close'])}")

# Load shedding actions
load_actions = df[df['action_type'].str.contains('load', na=False)]
print(f"\nTotal load control actions: {len(load_actions)}")
if len(load_actions) > 0:
    print(f"  Decreases: {len(load_actions[load_actions['action_type'] == 'load_decrease'])}")
    print(f"  Increases: {len(load_actions[load_actions['action_type'] == 'load_increase'])}")

# ============================================================================
# Question 2: How does topology change over time?
# ============================================================================
print("\n" + "="*80)
print("TOPOLOGY ANALYSIS")
print("="*80)

print(f"\nLive lines statistics:")
print(f"  Mean:    {df['live_lines'].mean():.1f}")
print(f"  Std:     {df['live_lines'].std():.1f}")
print(f"  Min:     {df['live_lines'].min():.0f}")
print(f"  Max:     {df['live_lines'].max():.0f}")

print(f"\nEnergized buses statistics:")
print(f"  Mean:    {df['energized_buses'].mean():.1f}")
print(f"  Std:     {df['energized_buses'].std():.1f}")
print(f"  Min:     {df['energized_buses'].min():.0f}")
print(f"  Max:     {df['energized_buses'].max():.0f}")

# Check if lines ever drop (fault occurs)
lines_dropped = df[df['live_lines'] < 37]
if len(lines_dropped) > 0:
    print(f"\n⚠️  Line faults detected in {lines_dropped['episode'].nunique()} episodes")
    print(f"   Total steps with <37 lines: {len(lines_dropped)}")
    print(f"   Minimum lines seen: {lines_dropped['live_lines'].min()}")
    
    # Find when lines drop
    for ep in lines_dropped['episode'].unique()[:3]:  # First 3 episodes with faults
        ep_data = df[df['episode'] == ep]
        fault_step = ep_data[ep_data['live_lines'] < 37].iloc[0]
        print(f"\n   Episode {ep}: Line fault at step {fault_step['step']}, "
              f"lines dropped from 37 to {fault_step['live_lines']}")
else:
    print("\n✓ No line faults detected - all episodes maintain 37 lines throughout")

# ============================================================================
# Question 3: Load management performance
# ============================================================================
print("\n" + "="*80)
print("LOAD MANAGEMENT ANALYSIS")
print("="*80)

print(f"\nCritical load preservation:")
print(f"  Mean:  {df['served_critical_frac'].mean()*100:.1f}%")
print(f"  Min:   {df['served_critical_frac'].min()*100:.1f}%")
print(f"  Max:   {df['served_critical_frac'].max()*100:.1f}%")

print(f"\nImportant load preservation:")
print(f"  Mean:  {df['served_important_frac'].mean()*100:.1f}%")
print(f"  Min:   {df['served_important_frac'].min()*100:.1f}%")
print(f"  Max:   {df['served_important_frac'].max()*100:.1f}%")

print(f"\nTotal load served:")
print(f"  Mean:  {df['served_total_frac'].mean()*100:.1f}%")
print(f"  Min:   {df['served_total_frac'].min()*100:.1f}%")
print(f"  Max:   {df['served_total_frac'].max()*100:.1f}%")

print(f"\nLoad shed (MW):")
print(f"  Mean:  {df['shed_mw'].mean():.2f} MW")
print(f"  Max:   {df['shed_mw'].max():.2f} MW")

# ============================================================================
# Question 4: DER constraint compliance
# ============================================================================
print("\n" + "="*80)
print("DER CONSTRAINT ANALYSIS")
print("="*80)

print(f"\nDER deficit (episode average):")
der_deficit_mean = (1 - df.groupby('episode')['der_deficit_frac'].mean()).mean()
print(f"  Mean deficit: {der_deficit_mean*100:.1f}%")

print(f"\nDER vs Load:")
print(f"  Avg DER available: {df['der_available_mw'].mean():.2f} MW")
print(f"  Avg DER online:    {df['der_online_mw'].mean():.2f} MW")
print(f"  Avg DER bound:     {df['der_bound_mw'].mean():.2f} MW")
print(f"  Avg load served:   {df['served_mw'].mean():.2f} MW")
print(f"  Avg total load:    {df['total_load_mw'].mean():.2f} MW")

# Check constraint violations
over_bound = df[df['served_mw'] > df['der_bound_mw'] + 0.05]  # 50kW tolerance
if len(over_bound) > 0:
    print(f"\n⚠️  DER bound violations: {len(over_bound)} steps ({len(over_bound)/len(df)*100:.1f}%)")
else:
    print(f"\n✓ No DER bound violations detected")

# ============================================================================
# Question 5: Safety compliance
# ============================================================================
print("\n" + "="*80)
print("SAFETY ANALYSIS")
print("="*80)

v_violations = df[df['v_violations'] > 0]
print(f"\nVoltage violations:")
print(f"  Steps with violations: {len(v_violations)} / {len(df)} ({len(v_violations)/len(df)*100:.1f}%)")
if len(v_violations) > 0:
    print(f"  Max violations per step: {v_violations['v_violations'].max()}")

pf_failures = df[df['powerflow_success'] == 0]
print(f"\nPowerflow failures:")
print(f"  Failed steps: {len(pf_failures)} / {len(df)} ({len(pf_failures)/len(df)*100:.1f}%)")

masked_actions = df[df['action_masked'] == 1]
print(f"\nSafety mask activations:")
print(f"  Actions masked: {len(masked_actions)} / {len(df)} ({len(masked_actions)/len(df)*100:.1f}%)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING PLOTS...")
print("="*80)

# Select first 3 episodes for detailed plots
episodes_to_plot = df['episode'].unique()[:3]

for ep in episodes_to_plot:
    ep_data = df[df['episode'] == ep]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'Episode {ep} - Agent Behavior Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Topology over time
    ax = axes[0, 0]
    ax.plot(ep_data['step'], ep_data['live_lines'], linewidth=2, color='#2c3e50', label='Live lines')
    ax.axhline(37, color='gray', linestyle='--', alpha=0.5, label='Full topology')
    ax.set_ylabel('Live Lines')
    ax.set_title('Network Topology')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([30, 40])
    
    # Mark switch actions
    switch_ops = ep_data[ep_data['action_type'].str.contains('switch', na=False)]
    for _, row in switch_ops.iterrows():
        color = 'green' if 'close' in row['action_type'] else 'red'
        ax.axvline(row['step'], color=color, alpha=0.3, linestyle=':')
    
    # Plot 2: Load served by tier
    ax = axes[0, 1]
    ax.plot(ep_data['step'], ep_data['served_critical_frac']*100, linewidth=2, label='Critical', color='#e74c3c')
    ax.plot(ep_data['step'], ep_data['served_important_frac']*100, linewidth=2, label='Important', color='#f39c12')
    ax.plot(ep_data['step'], ep_data['served_total_frac']*100, linewidth=2, label='Total', color='#3498db')
    ax.set_ylabel('Load Served (%)')
    ax.set_title('Load Preservation by Tier')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Plot 3: DER constraint tracking
    ax = axes[1, 0]
    ax.plot(ep_data['step'], ep_data['served_mw'], linewidth=2, label='Load served', color='#3498db')
    ax.plot(ep_data['step'], ep_data['der_bound_mw'], linewidth=2, label='DER bound', color='#e74c3c', linestyle='--')
    ax.plot(ep_data['step'], ep_data['der_online_mw'], linewidth=1.5, label='DER online', color='#2ecc71', alpha=0.7)
    ax.fill_between(ep_data['step'], 0, ep_data['der_bound_mw'], alpha=0.1, color='#e74c3c')
    ax.set_ylabel('Power (MW)')
    ax.set_title('DER Constraint Compliance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Voltage profile
    ax = axes[1, 1]
    ax.plot(ep_data['step'], ep_data['v_min_pu'], linewidth=1.5, label='V min', color='#e74c3c')
    ax.plot(ep_data['step'], ep_data['v_max_pu'], linewidth=1.5, label='V max', color='#3498db')
    ax.axhline(0.95, color='r', linestyle=':', alpha=0.5, label='Limits')
    ax.axhline(1.05, color='r', linestyle=':', alpha=0.5)
    ax.fill_between(ep_data['step'], 0.95, 1.05, alpha=0.1, color='green')
    ax.set_ylabel('Voltage (pu)')
    ax.set_title('Voltage Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.90, 1.10])
    
    # Plot 5: Action timeline
    ax = axes[2, 0]
    actions = ep_data[ep_data['action_type'] != 'no_op']
    if len(actions) > 0:
        action_types = actions['action_type'].unique()
        colors = {'switch_open': 'red', 'switch_close': 'green', 
                 'load_decrease': 'orange', 'load_increase': 'blue'}
        for i, act_type in enumerate(action_types):
            act_steps = actions[actions['action_type'] == act_type]['step'].values
            ax.eventplot([act_steps], colors=[colors.get(act_type, 'gray')], 
                        lineoffsets=i+1, linelengths=0.4, label=act_type)
        ax.set_ylabel('Action Type')
        ax.set_title('Agent Actions Timeline')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim([0, len(action_types)+1])
    else:
        ax.text(0.5, 0.5, 'No actions taken (all no-op)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Agent Actions Timeline')
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 6: Reward accumulation
    ax = axes[2, 1]
    cumulative_reward = ep_data['reward'].cumsum()
    ax.plot(ep_data['step'], cumulative_reward, linewidth=2, color='#9b59b6')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Reward Accumulation')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/episode_{ep}_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: results/episode_{ep}_analysis.png")

# Aggregate plot across all episodes
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('Aggregate Performance Across All Episodes', fontsize=14, fontweight='bold')

# Plot 1: Live lines distribution
ax = axes[0, 0]
for ep in df['episode'].unique():
    ep_data = df[df['episode'] == ep]
    ax.plot(ep_data['step'], ep_data['live_lines'], alpha=0.3, linewidth=1)
ax.axhline(37, color='black', linestyle='--', linewidth=2, label='Full topology')
ax.set_ylabel('Live Lines')
ax.set_xlabel('Step')
ax.set_title('Network Topology (All Episodes)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Load served by tier (average)
ax = axes[0, 1]
avg_crit = df.groupby('step')['served_critical_frac'].mean() * 100
avg_imp = df.groupby('step')['served_important_frac'].mean() * 100
avg_tot = df.groupby('step')['served_total_frac'].mean() * 100
steps = df.groupby('step')['step'].first()

ax.plot(steps, avg_crit, linewidth=2, label='Critical', color='#e74c3c')
ax.plot(steps, avg_imp, linewidth=2, label='Important', color='#f39c12')
ax.plot(steps, avg_tot, linewidth=2, label='Total', color='#3498db')
ax.set_ylabel('Load Served (%)')
ax.set_xlabel('Step')
ax.set_title('Average Load Served')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 105])

# Plot 3: DER compliance
ax = axes[1, 0]
avg_served = df.groupby('step')['served_mw'].mean()
avg_bound = df.groupby('step')['der_bound_mw'].mean()
ax.plot(steps, avg_served, linewidth=2, label='Load served', color='#3498db')
ax.plot(steps, avg_bound, linewidth=2, label='DER bound', color='#e74c3c', linestyle='--')
ax.fill_between(steps, 0, avg_bound, alpha=0.1, color='#e74c3c')
ax.set_ylabel('Power (MW)')
ax.set_xlabel('Step')
ax.set_title('DER Constraint Compliance (Average)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Action type distribution
ax = axes[1, 1]
action_counts.plot(kind='bar', ax=ax, color='steelblue')
ax.set_ylabel('Count')
ax.set_xlabel('Action Type')
ax.set_title('Action Distribution (All Episodes)')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/aggregate_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: results/aggregate_analysis.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nKey Findings:")
print(f"  • Agent maintains {df['served_critical_frac'].mean()*100:.1f}% critical load")
print(f"  • Agent maintains {df['served_important_frac'].mean()*100:.1f}% important load")
print(f"  • Average total load served: {df['served_total_frac'].mean()*100:.1f}%")
print(f"  • Network topology: {df['live_lines'].mean():.1f} ± {df['live_lines'].std():.1f} lines active")
print(f"  • DER bound violations: {len(over_bound)} steps ({len(over_bound)/len(df)*100:.2f}%)")
print(f"  • Voltage violations: {len(v_violations)} steps ({len(v_violations)/len(df)*100:.2f}%)")
print(f"  • Most common action: {action_counts.index[0]} ({action_counts.iloc[0]} times)")
print("="*80)