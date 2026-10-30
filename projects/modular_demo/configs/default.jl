# Typed Julia configuration keeps component construction explicit and inspectable.
# All new API spatial quantities are metres; all times are seconds.
using Simulator

function demo_experiment(; field=AnalyticField((p,t) -> sin(p[1]/30) + cos(p[2]/40) + t/300),
        adaptive=false)
    domain = BoxDomain((0.0,0.0), (100.0,80.0))
    candidates = [(x,y) for x in 10.0:20.0:90.0 for y in 10.0:20.0:70.0]
    planner = adaptive ? UncertaintyPlanner(candidates) :
        WaypointPlanner(lawnmower_waypoints(domain; spacing=20.0, margin=5.0))
    Experiment(; domain, field, sensor=GaussianSensor(0.1),
        vehicle=SingleIntegrator(2.0; power_model=ConstantPower(20.0)),
        estimator=GridGP(0.0:25.0:100.0, 0.0:20.0:80.0;
            length_scale=35.0, process_variance_rate=0.001),
        planner, controller=WaypointTracker(1.5),
        initial_state=VehicleState((5.0,5.0); battery_wh=100.0))
end
function demo_config()
    SimulationConfig(; duration=60.0, dt=1.0, sensor_every=2.0, plan_every=1.0,
        seed=42, query_points=[(x,y) for x in 0.0:25.0:100.0 for y in 0.0:20.0:80.0])
end
