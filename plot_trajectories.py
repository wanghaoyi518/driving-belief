# plot_trajectories.py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import world
import time

def load_trajectory_data(filename='data/world1-trajectories.pickle'):
    """Load trajectory data from pickle file"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def calculate_relative_metrics(robot_states, human_states):
    """Calculate relative metrics between cars"""
    # Longitudinal (y-axis) and lateral (x-axis) distances
    long_dist = human_states[:,1] - robot_states[:,1]
    lat_dist = human_states[:,0] - robot_states[:,0]
    
    # Absolute distance
    abs_dist = np.sqrt(long_dist**2 + lat_dist**2)
    
    # Relative velocity
    rel_vel = human_states[:,3] - robot_states[:,3]
    
    return long_dist, lat_dist, abs_dist, rel_vel

def plot_trajectories(robot_data, human_data):
    """Create multi-panel plot comparing robot and human trajectories"""
    # Create trajectory_plots directory if it doesn't exist
    plot_dir = "trajectory_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    robot_controls, robot_states, beliefs = robot_data[0]
    human_controls, human_states = human_data[0]  # Assuming same format without beliefs
    
    timesteps = len(robot_controls)
    time = np.arange(timesteps) * 0.1  # dt = 0.1
    
    # Calculate relative metrics
    long_dist, lat_dist, abs_dist, rel_vel = calculate_relative_metrics(robot_states, human_states)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. X-Y Trajectory Plot (Top Left)
    ax1 = plt.subplot(321)
    
    # Plot car positions at each timestep
    for i in range(timesteps):
        # Robot position (blue)
        ax1.plot(robot_states[i,0], robot_states[i,1], 'b.', alpha=0.3, markersize=2)
        # Human position (red)
        ax1.plot(human_states[i,0], human_states[i,1], 'r.', alpha=0.3, markersize=2)
        
        # Connect positions at same timestep with dotted line
        if i % 10 == 0:  # Every 10th step for clarity
            ax1.plot([robot_states[i,0], human_states[i,0]], 
                    [robot_states[i,1], human_states[i,1]], 
                    'k:', alpha=0.2)
    
    # Plot full trajectories
    ax1.plot(robot_states[:,0], robot_states[:,1], 'b-', label='Robot', linewidth=1)
    ax1.plot(human_states[:,0], human_states[:,1], 'r-', label='Human', linewidth=1)
    
    # Draw lane boundaries
    ax1.axvline(x=-0.13, color='k', linestyle='--', alpha=0.5)  # Center lane
    ax1.axvline(x=0.13, color='k', linestyle='--', alpha=0.5)   # Lane boundary
    ax1.axvline(x=-0.39, color='k', linestyle='--', alpha=0.5)  # Lane boundary
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Vehicle Trajectories')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # 2. Controls Plot (Top Right)
    ax2 = plt.subplot(322)
    # Robot controls
    ax2.plot(time, robot_controls[:,0], 'b-', label='Robot Steering', linewidth=1)
    ax2.plot(time, robot_controls[:,1], 'b--', label='Robot Acceleration', linewidth=1)
    # Human controls
    ax2.plot(time, human_controls[:,0], 'r-', label='Human Steering', linewidth=1)
    ax2.plot(time, human_controls[:,1], 'r--', label='Human Acceleration', linewidth=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control Input')
    ax2.set_title('Control Signals')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Relative Distances (Middle Left)
    ax3 = plt.subplot(323)
    ax3.plot(time, long_dist, 'g-', label='Longitudinal Distance', linewidth=1)
    ax3.plot(time, lat_dist, 'b-', label='Lateral Distance', linewidth=1)
    ax3.plot(time, abs_dist, 'r-', label='Absolute Distance', linewidth=1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Distance (m)')
    ax3.set_title('Relative Distances')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Velocities (Middle Right)
    ax4 = plt.subplot(324)
    ax4.plot(time, robot_states[:,3], 'b-', label='Robot Velocity', linewidth=1)
    ax4.plot(time, human_states[:,3], 'r-', label='Human Velocity', linewidth=1)
    ax4.plot(time, rel_vel, 'g--', label='Relative Velocity', linewidth=1)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Velocity Profiles')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Belief Evolution (Bottom Left)
    ax5 = plt.subplot(325)
    ax5.plot(time, beliefs[:,0], 'g-', label='P(Attentive)', linewidth=2)
    ax5.plot(time, beliefs[:,1], 'r-', label='P(Distracted)', linewidth=2)
    
    # Add markers for probing actions
    for i in range(len(robot_controls)):
        if abs(robot_controls[i,0]) > 0.05:  # Steering probe
            ax5.axvline(x=time[i], color='b', alpha=0.2)
        if robot_controls[i,1] < -0.05:  # Braking probe
            ax5.axvline(x=time[i], color='r', alpha=0.2)
            
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Probability')
    ax5.set_title('Belief Evolution with Probing Actions')
    ax5.legend()
    ax5.grid(True)
    
    # 6. Interaction Analysis (Bottom Right)
    ax6 = plt.subplot(326)
    # Plot human response delay
    response_delay = np.correlate(robot_controls[:,0], human_controls[:,0], mode='full')
    lags = np.arange(-(len(time)-1), len(time))
    ax6.plot(lags*0.1, response_delay, 'b-', label='Steering Response Correlation')
    ax6.set_xlabel('Lag (s)')
    ax6.set_ylabel('Response Correlation')
    ax6.set_title('Human Response Analysis')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    
    # Save plot with timestamp and analysis suffix
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"trajectory_plots/trajectory_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

def print_interaction_analysis(robot_data, human_data):
    """Print analysis of interactions between vehicles"""
    robot_controls, robot_states, beliefs = robot_data[0]
    human_controls, human_states = human_data[0]
    
    long_dist, lat_dist, abs_dist, rel_vel = calculate_relative_metrics(robot_states, human_states)
    
    print("\nInteraction Analysis:")
    print(f"Minimum separation distance: {np.min(abs_dist):.2f}m")
    print(f"Maximum relative velocity: {np.max(np.abs(rel_vel)):.2f}m/s")
    
    # Detect probing actions and human responses
    probe_times = []
    probe_types = []
    responses = []
    
    for t in range(len(robot_controls)-1):
        if abs(robot_controls[t,0]) > 0.05:  # Steering probe
            probe_times.append(t * 0.1)
            probe_types.append('Lane Nudge')
            # Look at human response in next 1 second
            response = np.max(np.abs(human_controls[t:t+10,0]))
            responses.append(response)
        elif robot_controls[t,1] < -0.05:  # Braking probe
            probe_times.append(t * 0.1)
            probe_types.append('Brake Test')
            response = np.min(human_controls[t:t+10,1])
            responses.append(response)
    
    print("\nProbing Actions and Responses:")
    for t, p, r in zip(probe_times, probe_types, responses):
        print(f"Time {t:.1f}s: {p}")
        print(f"  Maximum human response: {r:.3f}")
        idx = int(t/0.1)
        if idx < len(beliefs)-1:
            belief_change = beliefs[idx+1,1] - beliefs[idx,1]
            print(f"  Belief change in P(Distracted): {belief_change:.3f}")

def simulate_world(world_instance, n_steps=50):
    """Run simulation without visualization"""
    for _ in range(n_steps):
        for car in world_instance.cars:
            car.move()
    return world_instance

def plot_results(world_instance):
    """Plot final trajectory results"""
    plt.figure(figsize=(10, 8))
    
    # Plot lanes
    for lane in world_instance.lanes:
        plt.plot([lane.p[0], lane.q[0]], [lane.p[1], lane.q[1]], 'k--', alpha=0.5)
    
    # Get trajectory data
    human_pos = np.array(world_instance.position_history[0])
    robot_pos = np.array(world_instance.position_history[1])
    
    # Plot trajectories
    if len(human_pos) > 0:
        plt.plot(human_pos[:, 0], human_pos[:, 1], 'r-', label='Human', linewidth=2)
    if len(robot_pos) > 0:
        plt.plot(robot_pos[:, 0], robot_pos[:, 1], 'y-', label='Robot', linewidth=2)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Vehicle Trajectories')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def main():
    # Create world (use world1_active configuration)
    test_world = world.world1_active()
    
    # Run simulation without visualization
    simulated_world = simulate_world(test_world, n_steps=50)
    
    # Plot results
    plot_results(simulated_world)

if __name__ == "__main__":
    main()