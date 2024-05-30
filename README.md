# adaptive-ocean-exploration-sim

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


