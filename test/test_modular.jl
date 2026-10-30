using Test, Random, LinearAlgebra, Simulator

@testset "Data sources and interpolation" begin
    analytic = AnalyticField((p,t) -> 2p[1] - p[2] + 3t)
    xs,ys,ts = [0.,2.], [0.,4.], [0.,6.]
    grid = GriddedField(xs,ys,ts,[field_value(analytic,(x,y),t) for x in xs,y in ys,t in ts])
    @test field_value(grid,(.3,1.7),2.4) ≈ field_value(analytic,(.3,1.7),2.4)
    @test field_value(grid,(2.,4.),6.) == 18.
    @test_throws DomainError field_value(grid,(-.01,0.),0.)
    @test_throws DomainError field_value(grid,(0.,0.),7.)
    @test_throws ArgumentError GriddedField([0.,0.],ys,ts,zeros(2,2,2))
    @test_throws DimensionMismatch GriddedField(xs,ys,ts,zeros(2,2,1))
    # Constructor ownership: modifying source arrays cannot alter the field.
    values = zeros(2,2,2); owned = GriddedField(xs,ys,ts,values)
    values[1] = 100.; xs[1] = -100.
    @test field_value(owned,(0.,0.),0.) == 0.
end

@testset "Physical units, boundaries, speed, and energy" begin
    domain = BoxDomain((0.,0.),(10.,10.))
    vehicle = SingleIntegrator(2.; power_model=ConstantPower(360.))
    initial = VehicleState((1.,1.); battery_wh=10.)
    result = advance(vehicle, initial, (1.,0.), domain, 0., 2.5)
    @test result.position ≈ [3.5,1.]
    @test result.battery_wh ≈ 9.75
    # Stop at first boundary along a diagonal, preserving the commanded segment.
    clipped = advance(vehicle, VehicleState((9.,8.)), (2.,2.), domain, 0., 2.)
    @test clipped.position ≈ [10.,9.]
    limited = advance(vehicle, initial, (100.,0.), domain, 0., 1.)
    @test limited.position ≈ [3.,1.]
    @test_throws ArgumentError advance(vehicle, VehicleState((1.,1.); battery_wh=0.), (0.,0.), domain, 0., 1.)
    tracker = WaypointTracker(2.)
    @test control(tracker, initial, (1.1,1.), 0., 1.) ≈ [.1,0.]
end

@testset "GP adapter matches independent scalar Gaussian conditioning" begin
    gp = GridGP([0.],[0.]; sigma=2., length_scale=1., process_variance_rate=.5)
    b = initialize(gp, 0.)
    @test estimate(gp,b,(0.,0.)).variance ≈ 4.
    assimilate!(b,gp,Observation(0.,(0.,0.),3.; variance=1.))
    @test estimate(gp,b,(0.,0.)).mean ≈ 12/5
    @test estimate(gp,b,(0.,0.)).variance ≈ 4/5
    predict!(b,gp,2.)
    @test estimate(gp,b,(0.,0.)).variance ≈ 4/5 + 1
    @test_throws ArgumentError predict!(b,gp,1.)
    @test_throws ArgumentError assimilate!(b,gp,Observation(0.,(0.,0.),1.; variance=1.))
    @test_throws ArgumentError estimate(gp,b,(0.,0.),3.)
    # Existing square-root KF controlled prediction must use the state value.
    state = Simulator.KF.KFState(; μ=[1.], Σ=reshape([2.],1,1))
    predicted = Simulator.KF.predict(state, reshape([2.],1,1), reshape([3.],1,1), [4.], zeros(1,1))
    @test Simulator.KF.μ(predicted) == [14.]
end

function tiny_experiment(; field=AnalyticField((p,t)->p[1]+t/10), planner=WaypointPlanner([(0.,0.),(4.,0.)]))
    Experiment(; domain=BoxDomain((0.,0.),(5.,5.)), field,
        sensor=GaussianSensor(.1), vehicle=SingleIntegrator(1.),
        estimator=GridGP([0.,5.],[0.,5.]; length_scale=5.),
        planner, controller=WaypointTracker(1.), initial_state=VehicleState((0.,0.)))
end
@testset "Runner scheduling, independent state, and interchangeable components" begin
    cfg = SimulationConfig(; t0=10., duration=4., dt=.5, sensor_every=1., plan_every=1., seed=71,
        query_points=[(0.,0.),(5.,5.)])
    exp = tiny_experiment()
    result = run_experiment(exp; config=cfg)
    @test result.times == collect(10.:.5:14.)
    @test getproperty.(result.observations,:time) == collect(10.:14.)
    @test length(result.states) == length(result.commands)+1 == length(result.estimates)
    @test last(result.states).position ≈ [4.,0.]
    @test result.observations[3].position == result.states[5].position
    @test result.belief.time == last(result.times)
    # Reusing an experiment, even after unrelated runs, must reproduce noise/path.
    run_experiment(tiny_experiment(; planner=UncertaintyPlanner([(0.,0.),(5.,5.)])); config=cfg)
    again = run_experiment(exp; config=cfg)
    @test getproperty.(result.observations,:value) == getproperty.(again.observations,:value)
    @test getproperty.(result.states,:position) == getproperty.(again.states,:position)
    @test exp.planner.waypoints[1] == [0.,0.]
    # Private RNG: simulation must not consume the caller's default random stream.
    Random.seed!(93); expected = rand()
    Random.seed!(93); run_experiment(exp; config=cfg)
    @test rand() == expected
    # Exact linear sampled field can replace the analytic source unchanged.
    xs,ys,ts = [0.,5.],[0.,5.],[10.,14.]
    sampled = GriddedField(xs,ys,ts,[field_value(exp.field,(x,y),t) for x in xs,y in ys,t in ts])
    alternate = run_experiment(tiny_experiment(; field=sampled); config=cfg)
    @test getproperty.(alternate.observations,:value) ≈ getproperty.(result.observations,:value)
    # RMSE compares each snapshot with contemporaneous truth, not final truth.
    expected_rmse = [sqrt(sum((e.mean-field_value(exp.field,p,t))^2 for (e,p) in zip(es,cfg.query_points))/2)
        for (t,es) in zip(result.times,result.estimates)]
    @test field_rmse(result,exp.field) ≈ expected_rmse
    @test_throws ArgumentError SimulationConfig(; duration=1.,dt=.3)
    @test_throws ArgumentError SimulationConfig(; duration=1.,dt=.5,sensor_every=.75)
end

@testset "Recorded observation replay without truth or a vehicle" begin
    # A small CSV fixture is tracked; full datasets remain outside Git.
    lines = readlines(joinpath(@__DIR__,"fixtures","observations.csv"))[2:end]
    records = map(lines) do line
        t,x,y,value,variance = parse.(Float64,split(line,','))
        Observation(t,(x,y),value; variance)
    end
    gp = GridGP([0.,10.],[0.,10.]; length_scale=10., process_variance_rate=.1)
    replay = replay_observations(gp,records)
    manual = initialize(gp,first(records).time)
    for reading in records
        predict!(manual,gp,reading.time)
        assimilate!(manual,gp,reading)
    end
    @test replay.time == 4.
    @test estimate(gp,replay,(3.,4.)) == estimate(gp,manual,(3.,4.))
    @test_throws ArgumentError replay_observations(gp,reverse(records))
    @test_throws ArgumentError replay_observations(gp,Observation[])
end

@testset "Planner receives belief only" begin
    domain = BoxDomain((0.,0.),(10.,10.))
    context = PlanningContext(0., VehicleState((0.,0.)), domain,
        p -> FieldEstimate(0., p[1]))
    planner = UncertaintyPlanner([(1.,1.),(9.,9.)])
    @test plan!(initialize(planner,0.),planner,context) == [9.,9.]
    @test !hasproperty(context,:field)
    @test !hasproperty(context,:sensor)
end
