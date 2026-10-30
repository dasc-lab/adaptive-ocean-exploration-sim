# adaptive-ocean-exploration-sim

## Dependencies
To add `SpatiotemporalGPs.jl`, open the Julia REPL.
```
]
activate .
add https://github.com/dev10110/SpatiotemporalGPs.jl
```
this will install the SpatiotemporalGPs.jl package.

## Objectives:

Boat is exploring the ocean, and collecting `wind_x, wind_y`, and (in the future) salinity measurements. 

Objective is to study various control and measurement fusion algorithms. 

Languages: Pure julia implementation

Specifications:
- Single agent
- Boat is modeled in the simulation as a unicycle. No disturbances due to water (for now).
- Boat can be modelled in whatever form the controllers want for the control design.

- Mission Domain: Lake Jordan (approximately 10km x 10km)
- boat speed:
- mission duration: ~ 8hours
- update the interfaces to handle battery and solar
- deltaT for simulation needs to be set: 1 minute

## Update Rates
- Ergodic descent direction is computed every 60 seconds
- Ergodic velocity vector (based on Kavin's speed profile) is computed every 60 seconds

This results in a ergodic velocity vector (this is x,y velocities).

We feed this into the boundary avoidance system (Force Field), which generates the final control velocity vector every 1 seconds (need to tune maybe).

Measurements are collected at the rate at which the sensors provide them (can be on the order of 1-60 Hz).

Measurements have a location and value associated with them. 
We queue sensor data in a buffer.
Every 5 seconds, we average the values of the sensor data to generate a single measurement packet (Time, Location, Wind Speeds, SOC, etc).

Every 300 seconds (5 minutes);
- Hyperaparmeters are re-estimated every 300 seconds (5 minutes)
- Kalman Filter update/new resource estimate is recomputed every 300 seconds (5 minutes) (Maybe we should run the KF update more frequently than the HP updates?)

## Interfaces:

### Controller
```
struct BoatState
  position_x
  position_y
  heading
  # TODO: update to include battery state
end
```

```
struct Waypoint
  times::Vector{Float64}
  positions_x::Vector{Float64}
  positions_y::Vector{Float64}
end
```

```
struct ControlInput
  forward_velocity (m/s)
  heading_angle (rad, relative to +x axis, counter-clockwise)
end
```

```
struct InformationEstimate
  grid::Grid
  wind_x::Matrix
  wind_y::Matrix
end
```

```
struct UncertaintyMap
  grid::Grid
  wind_x_unc::Matrix
  wind_y_unc::Matrix
end
```



Each `Controller` needs to provide
```
# define the free space to operate in
Controller.set_mission_domain(mission_domain)

waypoints = Controller.trajectoryGenerator(current_time, current_position)
control_input = Controller.trackTrajectory(current_time, current_position, current_waypoints)
void Controller.fuseMeasurement(current_time, current_position, current_measurement)
information_map = Controller.information_map()
uncertainty_map = Controller.uncertainty_map()
```

### Simulator

```
struct Grid
  origin
  dxs
  Ls
end
```

```
ind_x, ind_y = GridWorld.pos2ind(pos_x, pos_y)
pos_x, pos_y = GridWorld.ind2pos(ind_x, ind_y)
```

```
struct Measurement
  measurement_time
  wind_x
  wind_y
end
```

```
struct MissionDomain
  shapeFile
end
```

`Simulator` needs to provide
```
new_boat_state = Simulator.step(current_time, current_boat_state, contol_input)
measurement = Simulator.measure(time, position) # running about every minute
```

The simulators can either be based on synthetic data, or based on real wind data. 
Simulator should detect boat leaving mission domain,


### Utilities

```
lat, lon = pos2coord(pos_x, pos_y)
pos_x, pos_y = coord2pos(lat, lon)
```


