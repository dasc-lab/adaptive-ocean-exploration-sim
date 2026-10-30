# Component architecture

The package remains named `Simulator` for compatibility. Use `using Simulator`
in new experiments. Ordinary source files share the package namespace; a folder
does not imply another package or require another module. Existing numerical
implementations retain their KF, NGPKF, and other historical module namespaces.

## Units and ownership

The new API uses local Cartesian **metres**, **seconds**, **m/s**, **watts**, and
battery **Wh**. Field values have dataset-defined units; observation variance
is in squared field units. Document the coordinate origin/frame in each study.
Historical code commonly uses kilometres/minutes/hours and is not automatically
converted. Never pass legacy coordinates into the new API without conversion.

Component objects describe a configuration. `run_experiment` deep-copies its
estimator, planner, controller, vehicle and sensor, then initializes fresh belief
and planner state. Fields are treated as read-only; analytic callbacks must be
pure. Random draws use a run-local MersenneTwister seeded by SimulationConfig.
Reproducibility assumes the same Julia/dependency versions and input data.

## Interfaces

| Component | Methods to implement | Contract |
|---|---|---|
| Domain | `in_domain(domain, p)`, `constrain_motion(domain, p, displacement)` | Return membership / safe position along a whole motion segment. |
| Field | `field_value(field, p, t)` | Scalar environmental value at the supplied position/time. No estimator dependency. |
| Sensor | `observe(sensor, field, vehicle_state, t, rng)` | Return an Observation with acquisition time, location, value and variance. |
| Vehicle | `advance(vehicle, state, command, domain, t, dt)` | Return next VehicleState. Enforce dynamics and domain constraints. |
| Power model | `power_watts(model, state, velocity, t)` | Electrical demand in W; negative values represent charging. |
| Estimator | `initialize(estimator,t0)`, `predict!(belief,estimator,t)`, `assimilate!(belief,estimator,observation)`, `estimate(estimator,belief,p,t)` | Own no truth. Return FieldEstimate(mean, variance). Prediction precedes assimilation at the same time. |
| Planner | `initialize(planner,t0)`, `plan!(state,planner,context)` | Return a target understood by the paired controller. Context exposes belief queries, vehicle state, domain and time. |
| Controller | `control(controller,vehicle_state,target,t,dt)` | Return a command understood by the paired vehicle. |

Subtyping the relevant abstract type and defining methods extends the package.
Use qualified definitions such as `Simulator.field_value(...)` or explicitly
import the function before extending it. Concrete components need not share
internal storage. Current bundled components use scalar fields, planar states,
waypoint targets and planar velocity commands. New target/command formats can
be added in compatible planner/controller/vehicle combinations.

## Scheduling

All cadences are integer multiples of the integration step. At each tick:

1. Predict the belief to the tick's timestamp.
2. Acquire and assimilate an observation if its sensor tick is due.
3. Save estimates at requested query points.
4. Replan if due, then compute a control command.
5. Move through one integration step.

Initial sensing/planning occurs at t0. The terminal tick performs prediction,
scheduled sensing and estimate saving, but no additional motion or planning.
There is no batching or implicit final observation. Positions/states have one
more entry than commands. Saved estimates share the state timestamps.

The single-integrator speed limit is enforced by the vehicle. Box-domain motion
stops at the first boundary intersection along the command segment. Insufficient
battery energy raises an error before returning a partially completed step.
This is a minimal offline runner, not a live vehicle driver.

## Included components

- BoxDomain; estimator-independent AnalyticField and trilinear GriddedField.
- GaussianSensor and chronological observation replay at irregular timestamps.
- SingleIntegrator with an explicit replaceable ConstantPower energy model.
- GridGP adapter over the existing square-root KF / numerical GP implementation.
  Random-walk process variance is per second; this is not a temporal-kernel GP.
  Observation variance must be positive. GridGP uses dense grid covariance and
  is intended for small grids; it inherits the existing algorithm's numerical
  limitations and matrix-inverse construction.
- WaypointPlanner, rectangular lawnmower waypoint generator, UncertaintyPlanner,
  and a WaypointTracker with exact one-step arrival.
- Contemporaneous field RMSE, separated from planning; optional trajectory CSV.

GriddedField accepts already loaded arrays, validates axes, copies input arrays,
and rejects extrapolation. File-format loaders and geographic transformations
belong in dataset adapters. Recorded observations can instead be replayed with
`replay_observations(estimator, readings)` without reconstructing a truth field.
No full-field truth metric is implied for field measurements alone.

## Legacy compatibility and migration

Existing implementation files are grouped under domains/, estimators/, planners/,
vehicles/ and legacy/. The old paths under src/ are compatibility includes, so
archived runners, notebooks and historical source-snapshot tooling still work.
Files use relative includes which work from either entry path. Compatibility
modules retain their original API, units, and Jordan Lake assumptions.

The new runner uses the GridGP adapter. Existing ergodic, STGP, SOC and transect
implementations remain callable through their historical modules but are not
all adapters to the new runner yet. Migrate them one at a time with analytical
or regression checks. The Jordan Lake domain, date/latitude globals and old
simulation loops remain isolated in legacy/ rather than entering new interfaces.
The original package-qualified Controller, Utilities, SyntheticData and
Measurement entry points remain available for older callers.

A study's configs construct components; experiments run them; analysis consumes
results. Keep notebooks thin. Put reusable behavior in src/ only after its inputs
and meaning are independent of the study. Each published run should record its
configuration, seed, dataset/checksum, code revision and dependency environment.
Existing dependencies are retained for compatibility; separating plotting and
study environments can follow as their legacy imports are migrated.
