# IMU Sensor Simulation and Noise Modeling Using Allan Variance

## Overview
This project implements a physics-based IMU (accelerometer + gyroscope) measurement simulator that converts true spacecraft/robot states and applied accelerations into realistic sensor outputs. The simulator injects stochastic noise components—including white noise, bias drift, and random walk—tuned using Allan variance parameters to replicate real inertial sensor behavior.
The tool is designed for navigation, GNC algorithm testing, and sensor characterization workflows where realistic accelerometer and gyro data are required.

## Key Concepts
-IMU measurement modeling (accelerometer + gyroscope)
-Coordinate-frame transformations and gravity projection
-Bias drift, white noise, and random-walk processes
-Allan variance and Allan deviation curve reproduction
-Synthetic IMU data generation
-Sensor characterization for navigation and GNC
-Simulation of measurement errors for EKF/UKF testing
-Noise parameter tuning to match specific sensor grades

## What I Did
-Developed a complete IMU simulator in Python to convert true kinematic states into synthetic sensor measurements
-Modeled accelerometer and gyro outputs including gravity, rotational dynamics, and applied forces
-Implemented stochastic noise models: white noise, bias drift, and random walk
-Tuned noise parameters using Allan variance to match realistic IMU characteristics
-Added tools to generate synthetic logs and compute Allan deviation curves
-Created a modular structure so the simulator can be integrated into navigation and control pipelines
-Tested sensor outputs to ensure realistic drift, long-term instability, and noise growth behavior

## How to Run
python examples/run_imu_simulation.py
python examples/generate_allan_deviation.py
