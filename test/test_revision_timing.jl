using Test, LinearAlgebra, Statistics, Random, StaticArrays, SpatiotemporalGPs
# Load the actual runner/controller definitions, without workers or main().
original_args = copy(ARGS)
empty!(ARGS)
append!(ARGS, ["--nworkers", "0"])
include(joinpath(@__DIR__, "..", "revision_sims", "run_half_domain_sim.jl"))
empty!(ARGS); append!(ARGS, original_args)

function small_case(; n=7, moving=false)
    dt=2.5/60
    ts=range(540.0; step=dt, length=n)
    gx=collect(0.4:0.1:0.5); gy=collect(0.4:0.1:0.5)
    points=vec([SVector(x,y) for x in gx,y in gy])
    ks=Matern(1/2,1.0,0.2); kt=Matern(1/2,1.0,0.2)
    problem=STGPKFProblem(points,ks,kt,dt)
    grid=NGPKF.NGPKFGrid(gx,gy,ks)
    env=(; xs=gx,ys=gy,ts,itp=(x,y,t)->2sin(20(t-540))+x-y)
    calls=Float64[]
    means=Matrix{Float64}[]
    controller=function(t,xs,mean,args...;kwargs...)
        push!(calls,t);push!(means,copy(mean))
        return [moving ? SVector(1.0,0.0) : SVector(0.0,0.0)],fill(.95,2,2)
    end
    return (; dt,ts,points,ks,problem,grid,env,calls,means,controller)
end
function simulate(c; fusion=5/60, control=c.dt, kwargs...)
    SimulatorST.simulate_known_param(c.ts,[SVector(.4,.4)],6000.,c.controller,
        fill(6000.,length(c.ts)),-3.5,JordanLakeDomain.convex_polygon,c.problem;
        ngpkf_grid=c.grid,EnvData=c.env,σ_meas=.5,
        fuse_measurements_every_ΔT=fusion,recompute_controller_every_ΔT=control,kwargs...)
end

@testset "Exact acquisition, prediction, and posterior timestamps" begin
    c=small_case(); Random.seed!(1234); r=simulate(c)
    @test r.measurement_ts == collect(c.ts)
    @test length(r.measurements)==length(c.ts)==length(r.measurement_positions)
    @test r.prediction_steps==length(c.ts)-1
    @test r.w_hat_ts ≈ collect(c.ts)[1:2:end]
    @test length(r.us)==length(r.speeds)==length(c.ts)-1
    @test length(r.bs)==length(r.xs)==length(c.ts)
    @test r.q_target_ts==c.calls==collect(c.ts)
    @test c.means[1] ≈ r.w_hats[1] # no artificial all-rated initial map
    @test !(all(c.means[1] .== -3.5))
    # Independent dense KF in physical field coordinates at grid observation 1.
    K=Matrix(SpatiotemporalGPs.STGPKF.kernel_matrix(c.ks,c.points,c.points))
    P=copy(K); μ=zeros(4); ρ=only(c.problem.ss_model.Φ)
    saved=1
    for i in eachindex(c.ts)
        if i>1
            μ=ρ*μ;P=ρ^2*P+(1-ρ^2)*K
        end
        gain=P[:,1]/(P[1,1]+.25)
        μ=μ+gain*(r.measurements[i]-μ[1])
        P=P-gain*P[1,:]'
        if isodd(i)
            @test vec(r.w_hats[saved]) ≈ μ atol=1e-8
            @test vec(r.ergo_q_maps[saved]) ≈ 1 ./ (1 .+ diag(P)) atol=1e-8
            saved+=1
        end
    end
    @test SpatiotemporalGPs.STGPKF.get_estimate(c.problem,r.filter_state) ≈ μ atol=1e-8
    # Exact sequential filtering is independent of the publication/fusion cadence.
    c2=small_case();Random.seed!(1234);r2=simulate(c2;fusion=c2.dt)
    @test r.measurements==r2.measurements
    @test r.w_hats[end] ≈ r2.w_hats[end] atol=1e-10
    @test r.ergo_q_maps[end] ≈ r2.ergo_q_maps[end] atol=1e-10
    @test r.bs[1]==6000
    expected=6000.
    for t in c.ts[1:end-1]
        expected=SoCController.batterymodel!(SoCController.ASV_Params(),SoCController.dayOfYear,t/60,SoCController.lat,0.,expected,c.dt/60)
    end
    @test r.bs[end] ≈ expected
end

@testset "Terminal flush and integer scheduling" begin
    c=small_case(n=6);r=simulate(c;control=5/60)
    @test r.w_hat_ts ≈ collect(c.ts)[[1,3,5,6]]
    @test r.q_target_ts ≈ collect(c.ts)[[1,3,5,6]]
    @test r.prediction_steps==5
    @test SimulatorST.cadence_steps(5/60,2.5/60)==2
    @test_throws ArgumentError SimulatorST.cadence_steps(6/60,2.5/60)
    @test_throws ArgumentError simulate(c;Q_process=I)
    wrong=merge(c,(;problem=STGPKFProblem(c.points,c.ks,Matern(1/2,1.,.2),c.dt*2)))
    @test_throws ArgumentError simulate(wrong)
end

@testset "Units, waypoint arrival, and west-half confinement" begin
    split_x=Transects.equal_area_split_x(JordanLakeDomain.convex_polygon)
    poly=Transects.west_half_polygon(JordanLakeDomain.convex_polygon,split_x)
    exact_west=Transects.west_half_polygon(JordanLakeDomain.convex_polygon,split_x;margin=0)
    full_area=Transects.polygon_area(JordanLakeDomain.convex_polygon.vertices)
    @test split_x ≈ 0.6525474113732039 atol=1e-12
    @test Transects.polygon_area(exact_west.vertices) ≈ full_area/2 atol=1e-12
    @test maximum(poly.vertices[1,:]) < split_x
    grid_pts=vec([[x,y] for x in .1:.3:2,y in .1:.3:2])
    pts=Transects.create_points_with_vertical_boundary(grid_pts,poly,.1:.3:2)
    @test !isempty(pts)
    @test all(p->p[1]<split_x && p in poly.polygon,pts)
    @test maximum(first,pts) >= split_x-3e-6
    @test count(p->p[1]>=split_x-3e-6,pts) >= 4
    c=small_case(n=2,moving=true);r=simulate(c)
    # 1 m/s for 2.5 s is 0.0025 km (old integration was 3.6x too slow).
    @test first(r.xs[2])[1]-first(r.xs[1])[1] ≈ .0025 atol=1e-12
    u,next=Transects.follow_waypoints([SVector(.4,.4)],[[.401,.4],[.6,.4]],1,1.,2.5/60;tolerance=0.)
    @test .4+first(u)[1]*(2.5/1000) ≈ .401
    @test next==1
    x=SVector(split_x-0.001,.5)
    u=SimulatorST.feasible_velocity(x,SVector(2.,0.),2.5/3600,poly,[0.,1.6],[0.,1.9])
    dest=x+u*2.5/1000
    @test dest[1] <= split_x-1e-6+1e-12
    @test norm(u) < 2.
    # A complete fast geometrical traversal must stay west, including loop closure.
    x=SVector(split_x-0.05,.75);index=1;visited=Set{Int}();max_x=x[1]
    for _ in 1:20000
        push!(visited,index)
        v,index=Transects.follow_waypoints([x],pts,index,2.,2.5/60)
        safe=SimulatorST.feasible_velocity(x,first(v),2.5/3600,poly,[0.,1.6],[0.,1.9])
        x=x+safe*2.5/1000
        max_x=max(max_x,x[1])
        @test x[1]<split_x
        @test x in poly.polygon
    end
    @test length(visited)==length(pts)
    @test max_x >= split_x-0.011
end

@testset "Ground-truth metrics use contemporaneous truth" begin
    env=(; synthetic_data=(;xs=[0.1,0.2],ys=[0.1,0.2]),w_rated_val=-3.5,
        convex_polygon=JordanLakeDomain.convex_polygon,
        ts_min=[0.,1.,2.],dt_min=1.)
    # Different truth slices make a one-step lag observable.
    maps=cat(fill(-3.5,2,2),fill(2.5,2,2),fill(-3.5,2,2);dims=3)
    env=merge(env,(;synthetic_data=merge(env.synthetic_data,(;data=maps))))
    r=(;w_hats=[maps[:,:,1],maps[:,:,2],maps[:,:,3]],w_hat_ts=[0.,1.,2.],
        ergo_q_maps=[fill(.5,2,2) for _ in 1:3],q_target_maps=[fill(.95,2,2)],q_target_ts=[0.])
    metrics=compute_run_metrics(r,env)
    @test metrics[1]==zeros(3)
    @test metrics[3] ≈ [.45,0.,.45]
    @test metrics[6][2] > .9 # stale planning target must not shift the truth time
end

@testset "Nonadaptive planning is uniform; evaluation target follows estimates" begin
    c=small_case()
    env=(; synthetic_data=c.env, Δt=2.5, convex_polygon=JordanLakeDomain.convex_polygon)
    grid=SimulatorST.ErgoGrid(c.grid,(2,2))
    achieved=fill(.5,2,2)
    x=[SVector(.4,.4)]
    for make_controller in (make_nonadaptive_ergo_controller,make_nonadaptive_bb_ipp_controller)
        left=make_controller(env);right=make_controller(env)
        # Swap estimates on successive calls while holding geometry and clarity fixed.
        # Stateful BB headings must evolve identically despite different estimates.
        for (i, (m1,m2)) in enumerate(((-3.5,2.5),(2.5,-3.5)))
            kwargs=(;ergo_grid=grid,ergo_q_map=achieved,traj=x,umax=1.,ΔT=c.dt)
            u1,q1=left(c.ts[i],x,fill(m1,2,2),-3.5,env.convex_polygon;kwargs...)
            u2,q2=right(c.ts[i],x,fill(m2,2,2),-3.5,env.convex_polygon;kwargs...)
            @test u1 ≈ u2
            @test q1 ≈ fill(.95*exp(-.25*(m1+3.5)^2),2,2)
            @test q2 ≈ fill(.95*exp(-.25*(m2+3.5)^2),2,2)
            @test q1 != q2
            @test mean(max.(q1-achieved,0.)) ≈ (m1 == -3.5 ? .45 : 0.)
        end
    end
end

@testset "Ergodic boundary steering preserves commanded speed" begin
    poly=JordanLakeDomain.convex_polygon
    center=vec(sum(poly.vertices;dims=2))/size(poly.vertices,2)
    positions=[SVector(.4,.4)]
    for i in axes(poly.vertices,2)
        a=poly.vertices[:,i];b=poly.vertices[:,mod1(i+1,size(poly.vertices,2))]
        push!(positions,SVector{2}(a))
        push!(positions,SVector{2}((a+b)/2))
        push!(positions,SVector{2}(.99a+.01center))
    end
    for p in positions, speed in (0.,.15,1.,1.8), angle in (0.,pi/2,pi,3pi/2)
        raw=speed*SVector(cos(angle),sin(angle))
        corrected=ErgodicController.convex_bounary_correction(poly,p,raw;
            speed_max=speed,preserve_speed=true)
        @test all(isfinite,corrected)
        @test norm(corrected) ≈ speed atol=1e-12
    end
    p=SVector(.4,.4)
    distance,closest=ErgodicController.ConvexBoundAvoidance.minimum_distance_to_boundary(poly,p)
    inward=ErgodicController.ConvexBoundAvoidance.normal_vector_to_centroid(poly,closest)
    raw=-1.3inward
    # Exact cancellation previously stopped the vehicle halfway through the buffer.
    corrected=ErgodicController.convex_bounary_correction(poly,p,raw;
        speed_max=1.3,min_safe_d=2distance,preserve_speed=true)
    @test corrected ≈ 1.3inward atol=1e-12
    # The default helper behavior remains unchanged for BB-IPP.
    @test norm(ErgodicController.convex_bounary_correction(poly,p,raw;
        speed_max=1.3,min_safe_d=2distance)) < 1e-12
    c=small_case();grid=SimulatorST.ErgoGrid(c.grid,(2,2))
    target=[.1 .9;.2 .8]
    for speed in (.15,1.,1.8)
        corrected=ErgodicController.controller_single_integrator_cvx_bound(
            grid,SVector(.43,.46),[SVector(.43,.46)],target,poly;umax=speed)
        @test norm(corrected) ≈ speed atol=1e-12
    end
end


@testset "Interpolation bounds tolerate roundoff only" begin
    xs = 0.0:0.05:0.1
    ys = 0.0:0.05:0.1
    ts = 540.0:0.5:541.0
    env = (; xs, ys, ts, itp=(x, y, t) -> x + 2y + 3t)
    Random.seed!(1234)
    expected = env.itp(0.05, 0.0, 540.5)
    @test SimulatorST.measure(env, 0.05, -4eps(Float64), 540.5, 0.0) == expected
    @test SimulatorST.measure(env, 0.05, 0.05, last(ts) + 4eps(last(ts)), 0.0) ==
        env.itp(0.05, 0.05, last(ts))
    @test_throws DomainError SimulatorST.measure(env, 0.05, -1e-9, 540.5, 0.0)
    @test_throws DomainError SimulatorST.measure(env, 0.11, 0.05, 540.5, 0.0)
end
