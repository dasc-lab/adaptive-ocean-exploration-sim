module Simulator

using LinearAlgebra, Random, StaticArrays

include("interfaces.jl")
include("domains/box.jl")
include("fields/synthetic/analytic.jl")
include("fields/datasets/gridded.jl")
include("sensors/gaussian.jl")
include("sensors/replay.jl")
include("vehicles/single_integrator.jl")

# Existing numerical algorithms, now shared by an explicit estimator adapter.
include("estimators/square_root_kf.jl")
include("estimators/ngpkf.jl")
include("estimators/grid_gp.jl")
include("planners/waypoints.jl")
include("planners/uncertainty.jl")
include("controllers/waypoint_tracker.jl")
include("simulation/experiment.jl")
include("simulation/runner.jl")
include("metrics/field_error.jl")
include("visualization/trajectory_csv.jl")

# Preserve qualified historical entry points. New experiments use the API above.
include("legacy/measurement.jl")
include("Controller.jl")
include("Utilities.jl")
include("SyntheticData.jl")
using .Controller, .Utilities, .SyntheticData

export AbstractDomain, AbstractField, AbstractSensor, AbstractVehicle,
    AbstractEstimator, AbstractPlanner, AbstractController, AbstractPowerModel
export Observation, FieldEstimate, VehicleState, PlanningContext
export field_value, in_domain, constrain_motion, observe, initialize,
    predict!, assimilate!, estimate, plan!, control, advance, power_watts
export BoxDomain, AnalyticField, GriddedField, GaussianSensor, replay_observations,
    ConstantPower, SingleIntegrator, GridGP, WaypointPlanner, UncertaintyPlanner,
    lawnmower_waypoints, WaypointTracker, Experiment, SimulationConfig,
    SimulationResult, run_experiment, field_rmse, write_trajectory_csv

end # module Simulator
