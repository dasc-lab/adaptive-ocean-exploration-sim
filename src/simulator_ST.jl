module SimulatorST

using LinearAlgebra, StaticArrays, Interpolations
using ProgressLogging
using SpatiotemporalGPs
using ..SyntheticData, ..NGPKF, ..ErgodicController, ..KF, ..SoCController, ..Variograms, ..JordanLakeDomain

using Plots

# MATERN SPATIAL LENGTH SCALE = 1.0 km
# MATERN TEMPORAL LENGTH SCALE = 0.2 * 60 * 24 = 4.8 hrs

struct MeasurementSpatial{T, P, F}
  t::T
  p::P
  y::F
end

"""
  measure(t, p, data; σ_meas = 0, Q_meas = σ_meas*I)

returns a MeasurementSpatial at time t and pos p by querying the data 
"""
function measure(t, p::SV, data::EDST; σ_meas=0, Q_meas=σ_meas * I) where {EDST<:EnvDataST, SV<:SVector{2}}
  y = data(p..., t) + (σ_meas * randn(1))[1]
  return MeasurementSpatial(t, p, y)
end

function measure(t, ps::VSV, data::EDST; σ_meas=0, Q_meas = σ_meas * I) where {EDST<:EnvDataST, SV<:SVector{2}, VSV <: AbstractVector{SV}}
  return [measure(t, p, data; σ_meas=σ_meas, Q_meas=Q_meas) for p in ps]
end

function measure(data, x, y, t, σ_m=0.1)
  return data.itp(x, y, t) + σ_m * randn()
end

function measure_reconstruction(data, t_idx)
  return data[t_idx]
end


function step(t, x::X, u::U, ΔT) where {X<:SVector, U<:SVector}
  A = I(2)
  B = ΔT * I(2)
  return A * x + B * u
end

function step(t, xs::XS, us::US, ΔT) where {X<:SVector, U<:SVector, XS<:AbstractVector{X}, US<:AbstractVector{U}}
  length(xs) == length(us) || throw(DimensionMismatch("xs and us must have the same length"))
  N = length(xs)
  return [step(t, xs[i], us[i], ΔT) for i = 1:N]
end

function step(t, xs::XS, u::U, ΔT) where {X<:SVector, U<:SVector, XS<:AbstractVector{X}}
  return [step(t, x, u, ΔT) for x in xs]
end

"""
Function to convert measurement arrays into MeasurementSpatial structs
"""
function data2struct(old_measurements, pos, ws, ts)
  for idx = 1:length(ts)
    push!(old_measurements, MeasurementSpatial(ts[idx], pos[idx], ws[idx]))
  end
  return old_measurements
end


struct SimResult{T,X,U,M,TV,W,EM}
  ts::T
  xs::X
  us::U
  measurements::M
  w_hat_ts::TV
  w_hats::W
  ergo_q_maps::EM
end

struct SimResultWeightedSpeed{T,X,U,S,B,M,TV,W,EM}
  ts::T
  xs::X
  us::U
  speeds::S
  bs::B
  measurements::M
  w_hat_ts::TV
  w_hats::W
  ergo_q_maps::EM
  q_target_maps
end

struct SimResultWeightedSpeedParams{T,X,U,SP,B,S,LS,LT,M,TV,W,EM}
  ts::T
  xs::X
  us::U
  speeds::SP
  bs::B
  σ_s::S
  λx_s::LS
  λt_s::LT
  measurements::M
  w_hat_ts::TV
  w_hats::W
  ergo_q_maps::EM
  q_target_maps
end

"""
Project position `x` back inside the domain bounds if integration overshoots.
"""
function clamp_to_domain(x::SVector{2, Float64}, xs::AbstractVector, ys::AbstractVector; eps=1e-4)
    x_clamped = clamp(x[1], xs[1] + eps, xs[end] - eps)
    y_clamped = clamp(x[2], ys[1] + eps, ys[end] - eps)
    return @SVector[x_clamped, y_clamped]
end

function clamp_to_domain(xs::AbstractVector{<:SVector}, domain_xs::AbstractVector, domain_ys::AbstractVector; eps=1e-4)
    return [clamp_to_domain(x, domain_xs, domain_ys; eps=eps) for x in xs]
end

function ErgoGrid(ngpkf_grid::G) where {G<:NGPKF.NGPKFGrid}
  origin = (ngpkf_grid.xs[1], ngpkf_grid.ys[1])
  dxs = (Base.step(ngpkf_grid.xs), Base.step(ngpkf_grid.ys))
  Ls = (maximum(ngpkf_grid.xs) - minimum(ngpkf_grid.xs), maximum(ngpkf_grid.ys) - minimum(ngpkf_grid.ys))

  ergo_grid = ErgodicController.Grid(origin, dxs, Ls)
  return ergo_grid
end

function ErgoGrid(ngpkf_grid::G, Ns) where {G<:NGPKF.NGPKFGrid}
  origin = (ngpkf_grid.xs[1], ngpkf_grid.ys[1])
  Ls = (maximum(ngpkf_grid.xs) - minimum(ngpkf_grid.xs), maximum(ngpkf_grid.ys) - minimum(ngpkf_grid.ys))
  dxs = Ls ./ (Ns .- 1)

  ergo_grid = ErgodicController.Grid(origin, dxs, Ls)
  return ergo_grid
end

function ngpkf_to_ergo(ngpkf_grid::G1, ergo_grid::G2, clarity_map) where {G1<:NGPKF.NGPKFGrid, G2<:ErgodicController.Grid}
  Ns = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  if Ns == ergo_grid.N
    return 1.0 * clarity_map
  end

  itp = linear_interpolation((ngpkf_grid.xs, ngpkf_grid.ys), clarity_map, extrapolation_bc=Line())
  ergo_map = itp(ErgodicController.xs(ergo_grid), ErgodicController.ys(ergo_grid))
  return ergo_map
end

### ALTERNATE CLARITY FORMULATION

C = 1
R = 0.5
Q = 0

clarity_with_decay(q, Q) = (C^2 / R) * (1-q)^2 - Q * q^2
clarity_only_decay(q, Q) = - Q * q^2

"""
rk4 dynamics for clarity with decay
"""
function clarity_dynamics_rk4_decay(q, h, Q; fn::Function = clarity_with_decay)
    f1 = fn(q, Q)
    f2 = fn(q + 0.5*h*f1, Q)
    f3 = fn(q + 0.5*h*f2, Q)
    f4 = fn(q + h*f3, Q)
    qn = q + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
    return qn
end

"""
clarity update function
"""
function update_clarity_decay_from_trajectories(grid, ϕ_curr, trajs, Q_mat, EnvData;
    footprint_cells=1, h=0.05)

    nx = length(EnvData.xs)
    ny = length(EnvData.ys)

    ϕ_new = copy(ϕ_curr)
    N_steps = length(trajs)

    @assert length(ϕ_curr) == nx*ny "ϕ_curr must have total length nx*ny"
    @assert size(Q_mat) == (nx, ny) "Q matrix must have size (nx, ny)"

    for k in 1:N_steps
      sensed_mask = falses(nx, ny)

      pt = trajs[k]
      positions = (pt isa SVector) ? (pt,) : pt

      for pos in positions
        current_x = pos[1]
        current_y = pos[2]

        xi = findmin(abs.(EnvData.xs .- current_x))[2]
        yi = findmin(abs.(EnvData.ys .- current_y))[2]

        row_range = max(1, xi - footprint_cells):min(nx, xi + footprint_cells)
        col_range = max(1, yi - footprint_cells):min(ny, yi + footprint_cells)

        for i in row_range, j in col_range
            sensed_mask[i, j] = true
        end
      end

      for j in 1:ny, i in 1:nx
        idx = i + (j-1)*nx
        local_Q = Q_mat[i, j]

        if sensed_mask[i, j]
          ϕ_new[idx] = clarity_dynamics_rk4_decay(ϕ_new[idx], h, local_Q; fn=clarity_with_decay)
        else
          ϕ_new[idx] = clarity_dynamics_rk4_decay(ϕ_new[idx], h, local_Q; fn=clarity_only_decay)
        end
      end
    end

    return ϕ_new
end
### END ALTERNATE CLARITY FORMULATION


# Simulate Spatial only 
function simulate_spatial(ts, x0::XS, controllers;
    ngpkf_grid::G,
    EnvDataST,
    σ_meas=0, 
    σ_process=0,
    Q_process = σ_process^2 * I,
    fuse_measurements_every_ΔT=5.0,
    recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector, XS<:AbstractVector{X}, G<:NGPKF.NGPKFGrid}
  
    t0 = ts[1]
    xs = [x0,]
    N_robots = length(x0)
    ΔT = Base.step(ts)
  
    ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))
  
    w_hat_ts = [t0,]
    w_hat = NGPKF.initialize(ngpkf_grid)
    w_hats = [w_hat,]

    q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
    ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
    ergo_q_maps = [ergo_q_map,]
  
    measurements = [measure(t0, x0, EnvDataST; σ_meas = σ_meas)...]  

    u0 = controllers(t0, x0;
      ngpkf_grid=ngpkf_grid,
      ergo_grid=ergo_grid,
      ergo_q_map=ergo_q_maps[end],
      traj=vcat(xs...),
      ΔT=ΔT,
    )
    us = [u0,]
  
    last_measurement_fuse_time = t0
    last_measurement_fuse_index = 0
  
    @progress for (it, t) in enumerate(ts[1:(end-1)])
      x = xs[end]

      ys = measure(t, x, EnvDataST; σ_meas=σ_meas)
      append!(measurements, ys)

      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT
        w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_w = [m.y for m in new_measurements]

        w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)
  
        q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)
  
        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end

      traj = vcat(xs...)
      u = controllers(t, x;
        ngpkf_grid=ngpkf_grid,
        ergo_grid=ergo_grid,
        ergo_q_map=ergo_q_maps[end],
        traj=traj,
        ΔT=ΔT,
      )
      push!(us, u)

      u = us[end]
      new_xs = step(t, xs[end], u, ΔT)
      new_xs = clamp_to_domain(new_xs, EnvDataST.xs, EnvDataST.ys)
      push!(xs, new_xs)
    end
  
    return SimResult(ts, xs, us, measurements, w_hat_ts, w_hats, ergo_q_maps)
end
  
# Simulate Spatiotemporal 
function simulate(ts, x0::XS, controllers;
  ngpkf_grid::G,
  EnvDataST,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector, XS<:AbstractVector{X}, G<:NGPKF.NGPKFGrid}

  t0 = ts[1]
  xs = [x0,]
  N_robots = length(x0)
  ΔT = Base.step(ts)
  
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  w_hat_ts = [t0,]

  wx_hat = NGPKF.initialize(ngpkf_grid)
  wx_hats = [wx_hat,]

  wy_hat = NGPKF.initialize(ngpkf_grid)
  wy_hats = [wy_hat,]

  q_map = NGPKF.clarity_map(ngpkf_grid, wx_hat, wy_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  measurements = [measure(t0, x0, EnvDataST; σ_meas = σ_meas)...]  

  u0 = controllers(t0, x0;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    ΔT=ΔT,
  )
  us = [u0,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  @progress for (it, t) in enumerate(ts[1:(end-1)])
    x = xs[end]
 
    ys = measure(t, x, EnvDataST; σ_meas=σ_meas)
    append!(measurements, ys)

    if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT
      wx_hat = NGPKF.predict(ngpkf_grid, wx_hats[end]; Q_process=Q_process)
      wy_hat = NGPKF.predict(ngpkf_grid, wy_hats[end]; Q_process=Q_process)

      new_measurements = measurements[(last_measurement_fuse_index+1):end]
      measurement_pos = [m.p for m in new_measurements]
      measurements_wx = [m.y[1] for m in new_measurements]
      measurements_wy = [m.y[2] for m in new_measurements]

      wx_hat_new = NGPKF.correct(ngpkf_grid, wx_hat, measurement_pos, measurements_wx; σ_meas=σ_meas)
      wy_hat_new = NGPKF.correct(ngpkf_grid, wy_hat, measurement_pos, measurements_wy; σ_meas=σ_meas)

      push!(w_hat_ts, t)
      push!(wx_hats, wx_hat_new)
      push!(wy_hats, wy_hat_new)

      q_map = NGPKF.clarity_map(ngpkf_grid, wx_hat_new, wy_hat_new)
      ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
      push!(ergo_q_maps, ergo_q_map)

      last_measurement_fuse_time = t
      last_measurement_fuse_index = length(measurements)
    end

    traj = vcat(xs...)
    u = controllers(t, x;
      ngpkf_grid=ngpkf_grid,
      ergo_grid=ergo_grid,
      ergo_q_map=ergo_q_maps[end],
      traj=traj,
      ΔT=ΔT,
    )
    push!(us, u)

    u = us[end]
    new_xs = step(t, xs[end], u, ΔT)
    new_xs = clamp_to_domain(new_xs, EnvDataST.xs, EnvDataST.ys)
    push!(xs, new_xs)
  end

  return SimResult(ts, xs, us, measurements, w_hat_ts, (wx_hats, wy_hats), ergo_q_maps)
end

# Spatiotemporal Jordan Lake Simulation with known hyperparameters
function simulate_known_param(ts, x0::XS, b0, controllers, soc_profile, w_rated_val, convex_polygon, stgp_problem;
  ngpkf_grid::G,
  EnvData,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0/60,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector, XS<:AbstractVector{X}, G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288;
  lat = 35.45;
  
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys) 
  ergo_grid = ErgoGrid(ngpkf_grid, (length(EnvData.xs), length(EnvData.ys)))

  w_hat_ts = [t0,]
  state = stgpkf_initialize(stgp_problem)
  est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
  w_hat = reshape(est, length(EnvData.xs), length(EnvData.ys))
  w_hats = [w_hat, ]

  qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
  q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  measurements = [measure(EnvData, x0[1]..., EnvData.ts[1], σ_meas)...]

  M = ones(Nx, Ny) * w_rated_val

  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0, soc_profile[1], error_sum, error);
  speeds = [speed];

  u0, q_target = controllers(t0, x0, M, w_rated_val, convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    umax=speed,
    ΔT=ΔT,
  )
  us = [u0,]

  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0/60, lat, norm(u0), b0, ΔT/60))
  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  @progress for (it, t) in enumerate(ts[1:(end-1)])
    if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT
      measurement_pos = vec([@SVector[xs[i][1][1], xs[i][1][2]] for i in (last_measurement_fuse_index+1):length(xs)])
      measurements_w = measurements[(last_measurement_fuse_index+1):end]
      measure_Σ = (σ_meas^2) * I(length(measurements_w))

      state_correction = stgpkf_correct(stgp_problem, state, measurement_pos, measurements_w, measure_Σ)
      state = stgpkf_predict(stgp_problem, state_correction)
      
      est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
      w_hat_new = reshape(est, length(EnvData.xs), length(EnvData.ys))

      push!(w_hat_ts, t)
      push!(w_hats, w_hat_new)

      qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
      q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
      ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
      push!(ergo_q_maps, ergo_q_map)

      last_measurement_fuse_time = t
      last_measurement_fuse_index = length(measurements)
    end
    
    x = xs[end]
    b = bs[end]

    ys = [measure(EnvData, x[end]..., EnvData.ts[it], σ_meas)...]
    append!(measurements, ys)

    traj = vcat(xs...)
    M = w_hats[end]

    speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);
    push!(speeds, speed);
    
    u, q_target = controllers(t, x, M, w_rated_val, convex_polygon;
      ngpkf_grid=ngpkf_grid,
      ergo_grid=ergo_grid,
      ergo_q_map=ergo_q_maps[end],
      traj=traj,
      umax=speed,
      ΔT=ΔT,
    )
    
    u = norm(u) > 1e-8 ? normalize(u) * speed : zero(u)

    push!(q_target_maps, q_target)
    push!(us, u)

    new_xs = step(t, xs[end], u, ΔT/60)
    new_xs = clamp_to_domain(new_xs, EnvData.xs, EnvData.ys)
    
    push!(xs, new_xs)
    push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u), b, ΔT/60))
  end

  return SimResultWeightedSpeed(ts, xs, us, speeds, bs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)
end

# Simulate a stationary boat
function simulate_stationary(ts, x0::XS, b0, controllers, soc_profile, w_rated_val, convex_polygon, stgp_problem;
  ngpkf_grid::G,
  EnvData,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0/60,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector, XS<:AbstractVector{X}, G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288;
  lat = 35.45;
  
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys) 
  ergo_grid = ErgoGrid(ngpkf_grid, (length(EnvData.xs), length(EnvData.ys)))

  w_hat_ts = [t0,]
  state = stgpkf_initialize(stgp_problem)
  est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
  w_hat = reshape(est, length(EnvData.xs), length(EnvData.ys))
  w_hats = [w_hat, ]

  qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
  q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  measurements = [measure(EnvData, x0[1]..., EnvData.ts[1], σ_meas)...]

  M = ones(Nx, Ny) * w_rated_val

  error = 0.0;
  error_sum = 0.0;
  speed = 0.01;
  speeds = [speed];

  u0, q_target = controllers(t0, x0, M, w_rated_val, convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    umax=speed,
    ΔT=ΔT,
  )
  us = [u0,]

  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0/60, lat, norm(u0), b0, ΔT/60))
  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  @progress for (it, t) in enumerate(ts[1:(end-1)])
    if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT
      measurement_pos = vec([@SVector[xs[i][1][1], xs[i][1][2]] for i in (last_measurement_fuse_index+1):length(xs)])
      measurements_w = measurements[(last_measurement_fuse_index+1):end]
      measure_Σ = (σ_meas^2) * I(length(measurements_w));

      state_correction = stgpkf_correct(stgp_problem, state, measurement_pos, measurements_w, measure_Σ)
      state = stgpkf_predict(stgp_problem, state_correction)
      
      est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
      w_hat_new = reshape(est, length(EnvData.xs), length(EnvData.ys))

      push!(w_hat_ts, t)
      push!(w_hats, w_hat_new)

      qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
      q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
      ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
      push!(ergo_q_maps, ergo_q_map)

      last_measurement_fuse_time = t
      last_measurement_fuse_index = length(measurements)
    end
    
    x = xs[end]
    b = bs[end]

    ys = [measure(EnvData, x[end]..., EnvData.ts[it], σ_meas)...]
    append!(measurements, ys)

    traj = vcat(xs...)
    M = w_hats[end]

    speed = 0.01;
    push!(speeds, speed);
    
    u, q_target = controllers(t, x, M, w_rated_val, convex_polygon;
      ngpkf_grid=ngpkf_grid,
      ergo_grid=ergo_grid,
      ergo_q_map=ergo_q_maps[end],
      traj=traj,
      umax=speed,
      ΔT=ΔT,
    )
    
    u = norm(u) > 1e-8 ? normalize(u) * speed : zero(u)

    push!(q_target_maps, q_target)
    push!(us, u)

    new_xs = step(t, xs[end], u, ΔT/60)
    new_xs = clamp_to_domain(new_xs, EnvData.xs, EnvData.ys)
    
    push!(xs, new_xs)
    push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u), b, ΔT/60))
  end

  return SimResultWeightedSpeed(ts, xs, us, speeds, bs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)
end

# Spatiotemporal Jordan Lake Simulation with known hyperparameters
function simulate_reconstruction(ts, x0::XS, b0, controllers, soc_profile, w_rated_val, convex_polygon, stgp_problem;
  ngpkf_grid::G,
  EnvData,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0/60,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector, XS<:AbstractVector{X}, G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288;
  lat = 35.45;
  
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys) 
  ergo_grid = ErgoGrid(ngpkf_grid, (length(EnvData.xs), length(EnvData.ys)))

  w_hat_ts = [t0,]
  state = stgpkf_initialize(stgp_problem)
  est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
  w_hat = reshape(est, length(EnvData.xs), length(EnvData.ys))
  w_hats = [w_hat, ]

  qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
  q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  LookUpData = EnvData.data[1, 1, :]
  measurements = [measure_reconstruction(LookUpData, 1)...]

  M = ones(Nx, Ny) * w_rated_val

  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0, soc_profile[1], error_sum, error);
  speeds = [speed];

  u0, q_target = controllers(t0, x0, M, w_rated_val, convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    umax=speed,
    ΔT=ΔT,
  )
  us = [u0,]

  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0/60, lat, norm(u0), b0, ΔT/60))
  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  @progress for (it, t) in enumerate(ts[1:(end-1)])
    if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT
      measurement_pos = vec([@SVector[xs[i][1][1], xs[i][1][2]] for i in (last_measurement_fuse_index+1):length(xs)])
      measurements_w = measurements[(last_measurement_fuse_index+1):end]
      measure_Σ = (σ_meas^2) * I(length(measurements_w));

      state_correction = stgpkf_correct(stgp_problem, state, measurement_pos, measurements_w, measure_Σ)
      state = stgpkf_predict(stgp_problem, state_correction)
      
      est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
      w_hat_new = reshape(est, length(EnvData.xs), length(EnvData.ys))

      push!(w_hat_ts, t)
      push!(w_hats, w_hat_new)

      qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
      q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
      ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
      push!(ergo_q_maps, ergo_q_map)

      last_measurement_fuse_time = t
      last_measurement_fuse_index = length(measurements)
    end
    
    x = xs[end]
    b = bs[end]

    ys = [measure_reconstruction(LookUpData, it)...]
    append!(measurements, ys)

    traj = vcat(xs...)
    M = w_hats[end]

    speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);
    push!(speeds, speed);
    
    u, q_target = controllers(t, x, M, w_rated_val, convex_polygon;
      ngpkf_grid=ngpkf_grid,
      ergo_grid=ergo_grid,
      ergo_q_map=ergo_q_maps[end],
      traj=traj,
      umax=speed,
      ΔT=ΔT,
    )

    u = norm(u) > 1e-8 ? normalize(u) * speed : zero(u)
    
    push!(q_target_maps, q_target)
    push!(us, u)

    new_xs = step(t, xs[end], u, ΔT)
    new_xs = clamp_to_domain(new_xs, EnvData.xs, EnvData.ys)

    push!(xs, new_xs)
    push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u), b, ΔT/60))
  end

  return SimResultWeightedSpeed(ts, xs, us, speeds, bs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)
end

# Spatiotemporal Jordan Lake Simulation with estimated parameters
function simulate_param_est(ts, x0::XS, b0, controllers, soc_profile, w_rated_val, convex_polygon, stgp_problem;
  ngpkf_grid::G,
  EnvData,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0/60,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector, XS<:AbstractVector{X}, G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288;
  lat = 35.45;

  σs = [1.0, ]
  λxs = [1.0, ]
  λts = [1.0, ]

  measure_vec = []

  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  w_hat_ts = [t0,]

  state = stgpkf_initialize(stgp_problem)
  est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
  w_hat = reshape(est, length(EnvData.xs), length(EnvData.ys))
  w_hats = [w_hat, ]

  qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
  q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  measurements = [measure(EnvData, x0[1]..., EnvData.ts[1], σ_meas)...]

  M = ones(Nx, Ny) * w_rated_val

  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0, soc_profile[1], error_sum, error);
  speeds = [speed];

  u0, q_target = controllers(t0, x0, M, w_rated_val, convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    umax=speed,
    ΔT=ΔT,
  )
  us = [u0,]

  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0/60, lat, norm(u0), b0, ΔT/60))
  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  @progress for (it, t) in enumerate(ts[1:(end-1)])
    if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT
      measurement_pos = vec([@SVector[xs[i][1][1], xs[i][1][2]] for i in (last_measurement_fuse_index+1):length(xs)])
      measurements_w = measurements[(last_measurement_fuse_index+1):end]
      measure_Σ = (σ_meas^2) * I(length(measurements_w));

      measure_vec = data2struct(measure_vec, measurement_pos, measurements_w, ts[it-length(measurements_w)+1:it])

      σt = 2.0   # m/s

      if length(measure_vec) > 1000
        σ, λx, λt = Variograms.hp_fit(measure_vec[end-1000:end])
      else
        σ, λx, λt = Variograms.hp_fit(measure_vec)
      end

      σ = max(σ, 0.001)
      λx = max(λx, 0.001)
      λt = max(λt, 0.001)
      push!(σs, σ)
      push!(λxs, λx)
      push!(λts, λt)

      kt = Matern(1/2, σt, λt)
      ks = Matern(1/2, σ, λx)

      Δx = 0.10 # km
      gridxs = 0:Δx:1.4
      gridys = 0:Δx:6.5

      grid_pts = vec([@SVector[x, y] for x in gridxs, y in gridys]);

      stgp_problem = STGPKFProblem(grid_pts, ks, kt, ΔT)
      kern = ks
      ngpkf_grid = NGPKF.NGPKFGrid(EnvData.xs, EnvData.ys, kern)
      ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))       

      state_correction = stgpkf_correct(stgp_problem, state, measurement_pos, measurements_w, measure_Σ)
      state = stgpkf_predict(stgp_problem, state_correction)
      
      est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
      w_hat_new = reshape(est, length(EnvData.xs), length(EnvData.ys))

      push!(w_hat_ts, t)
      push!(w_hats, w_hat_new)

      qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
      q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
      ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
      push!(ergo_q_maps, ergo_q_map)

      last_measurement_fuse_time = t
      last_measurement_fuse_index = length(measurements)
    end

    x = xs[end]
    b = bs[end]

    ys = [measure(EnvData, x[end]..., EnvData.ts[it], σ_meas)...]
    append!(measurements, ys)

    traj = vcat(xs...)
    M = w_hats[end]

    speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);
    push!(speeds, speed);
    
    u, q_target = controllers(t, x, M, w_rated_val, convex_polygon;
      ngpkf_grid=ngpkf_grid,
      ergo_grid=ergo_grid,
      ergo_q_map=ergo_q_maps[end],
      traj=traj,
      umax=speed,
      ΔT=ΔT,
    )
    
    push!(q_target_maps, q_target)

    if any(isnan, u)
      u = [@SVector[0.0, 0.0]]
    end
    push!(us, u)

    u_step = us[end]
    new_xs = step(t, xs[end], u_step, ΔT)
    new_xs = clamp_to_domain(new_xs, EnvData.xs, EnvData.ys)

    push!(xs, new_xs)
    push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u_step), b, ΔT/60))
  end
  
  return SimResultWeightedSpeedParams(ts, xs, us, speeds, bs, σs, λxs, λts, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)
end

function simulate_transect(ts, x0::XS, b0, controllers, soc_profile, w_rated_val, convex_polygon, stgp_problem;
  ngpkf_grid::G,
  EnvData,
  transect_pts,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0/60,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector, XS<:AbstractVector{X}, G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288;
  lat = 35.45;
  
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys) 
  ergo_grid = ErgoGrid(ngpkf_grid, (length(EnvData.xs), length(EnvData.ys)))

  w_hat_ts = [t0,]
  state = stgpkf_initialize(stgp_problem)
  est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
  w_hat = reshape(est, length(EnvData.xs), length(EnvData.ys))
  w_hats = [w_hat, ]

  qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
  q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  LookUpData = EnvData.data[1, 1, :]
  measurements = [measure_reconstruction(LookUpData, 1)...]

  M = ones(Nx, Ny) * w_rated_val

  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0, soc_profile[1], error_sum, error);
  speeds = [speed];

  waypoint_idx = 1

  u0, q_target, waypoint_idx = controllers(t0, x0, M, w_rated_val, convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    transect_pts=transect_pts,
    waypoint_idx=waypoint_idx,
    umax=speed,
    ΔT=ΔT,
  )
  us = [u0,]

  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0/60, lat, norm(u0), b0, ΔT/60))
  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  @progress for (it, t) in enumerate(ts[1:(end-1)])
    if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT
      measurement_pos = vec([@SVector[xs[i][1][1], xs[i][1][2]] for i in (last_measurement_fuse_index+1):length(xs)])
      measurements_w = measurements[(last_measurement_fuse_index+1):end]
      measure_Σ = (σ_meas^2) * I(length(measurements_w));

      state_correction = stgpkf_correct(stgp_problem, state, measurement_pos, measurements_w, measure_Σ)
      state = stgpkf_predict(stgp_problem, state_correction)
      
      est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
      w_hat_new = reshape(est, length(EnvData.xs), length(EnvData.ys))

      push!(w_hat_ts, t)
      push!(w_hats, w_hat_new)

      qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
      q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
      ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
      push!(ergo_q_maps, ergo_q_map)

      last_measurement_fuse_time = t
      last_measurement_fuse_index = length(measurements)
    end
    
    x = xs[end]
    b = bs[end]

    ys = [measure_reconstruction(LookUpData, it)...]
    append!(measurements, ys)

    traj = vcat(xs...)
    M = w_hats[end]

    speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);
    push!(speeds, speed);
    
    u, q_target, waypoint_idx = controllers(t, x, M, w_rated_val, convex_polygon;
      ngpkf_grid=ngpkf_grid,
      ergo_grid=ergo_grid,
      ergo_q_map=ergo_q_maps[end],
      traj=traj,
      transect_pts=transect_pts,
      waypoint_idx=waypoint_idx,
      umax=speed,
      ΔT=ΔT,
    )
    
    push!(q_target_maps, q_target)
    push!(us, u)

    u_step = us[end]
    new_xs = step(t, xs[end], u_step, ΔT/60)
    new_xs = clamp_to_domain(new_xs, EnvData.xs, EnvData.ys)

    push!(xs, new_xs)
    push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u_step), b, ΔT/60))
  end

  return SimResultWeightedSpeed(ts, xs, us, speeds, bs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)
end

function simulate_known_transect(ts, x0::XS, b0, controllers, soc_profile, w_rated_val, convex_polygon, stgp_problem;
  ngpkf_grid::G,
  EnvData,
  transect_pts,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0/60,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector, XS<:AbstractVector{X}, G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288;
  lat = 35.45;
  
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys) 
  ergo_grid = ErgoGrid(ngpkf_grid, (length(EnvData.xs), length(EnvData.ys)))

  w_hat_ts = [t0,]
  state = stgpkf_initialize(stgp_problem)
  est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
  w_hat = reshape(est, length(EnvData.xs), length(EnvData.ys))
  w_hats = [w_hat, ]

  qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
  q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  measurements = [measure(EnvData, x0[1]..., EnvData.ts[1], σ_meas)...]

  M = ones(Nx, Ny) * w_rated_val

  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0, soc_profile[1], error_sum, error);
  speeds = [speed];

  waypoint_idx = 1

  u0, q_target, waypoint_idx = controllers(t0, x0, M, w_rated_val, convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    transect_pts=transect_pts,
    waypoint_idx=waypoint_idx,
    umax=speed,
    ΔT=ΔT,
  )
  us = [u0,]

  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0/60, lat, norm(u0), b0, ΔT/60))
  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  @progress for (it, t) in enumerate(ts[1:(end-1)])
    if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT
      measurement_pos = vec([@SVector[xs[i][1][1], xs[i][1][2]] for i in (last_measurement_fuse_index+1):length(xs)])
      measurements_w = measurements[(last_measurement_fuse_index+1):end]
      measure_Σ = (σ_meas^2) * I(length(measurements_w));

      state_correction = stgpkf_correct(stgp_problem, state, measurement_pos, measurements_w, measure_Σ)
      state = stgpkf_predict(stgp_problem, state_correction)
      
      est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
      w_hat_new = reshape(est, length(EnvData.xs), length(EnvData.ys))

      push!(w_hat_ts, t)
      push!(w_hats, w_hat_new)

      qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
      q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
      ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
      push!(ergo_q_maps, ergo_q_map)

      last_measurement_fuse_time = t
      last_measurement_fuse_index = length(measurements)
    end
    
    x = xs[end]
    b = bs[end]

    ys = [measure(EnvData, x[end]..., EnvData.ts[it], σ_meas)...]
    append!(measurements, ys)

    traj = vcat(xs...)
    M = w_hats[end]

    speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);
    push!(speeds, speed);
    
    u, q_target, waypoint_idx = controllers(t, x, M, w_rated_val, convex_polygon;
      ngpkf_grid=ngpkf_grid,
      ergo_grid=ergo_grid,
      ergo_q_map=ergo_q_maps[end],
      traj=traj,
      transect_pts=transect_pts,
      waypoint_idx=waypoint_idx,
      umax=speed,
      ΔT=ΔT,
    )
    
    push!(q_target_maps, q_target)
    push!(us, u)

    u_step = us[end]
    new_xs = step(t, xs[end], u_step, ΔT/60)
    new_xs = clamp_to_domain(new_xs, EnvData.xs, EnvData.ys)

    push!(xs, new_xs)
    push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u_step), b, ΔT/60))
  end

  return SimResultWeightedSpeed(ts, xs, us, speeds, bs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)
end

function simulate_nongp_ergo(ts, x0::XS, b0, controllers, soc_profile, w_rated_val, convex_polygon, stgp_problem;
  ngpkf_grid::G,
  EnvData,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0/60,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector, XS<:AbstractVector{X}, G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288;
  lat = 35.45;
  
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys) 
  ergo_grid = ErgoGrid(ngpkf_grid, (length(EnvData.xs), length(EnvData.ys)))

  w_hat_ts = [t0,]
  state = stgpkf_initialize(stgp_problem)
  est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
  w_hat = reshape(est, length(EnvData.xs), length(EnvData.ys))
  w_hats = [w_hat, ]

  qs = zeros(length(EnvData.xs) * length(EnvData.ys))
  q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
  Q_mat = zeros(length(EnvData.xs), length(EnvData.ys))
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  measurements = [measure(EnvData, x0[1]..., EnvData.ts[1], σ_meas)...]

  M = ones(Nx, Ny) * w_rated_val

  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0, soc_profile[1], error_sum, error);
  speeds = [speed];

  u0, q_target = controllers(t0, x0, M, w_rated_val, convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    umax=speed,
    ΔT=ΔT,
  )
  us = [u0,]

  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0/60, lat, norm(u0), b0, ΔT/60))
  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  @progress for (it, t) in enumerate(ts[1:(end-1)])
    if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT
      measurement_pos = vec([@SVector[xs[i][1][1], xs[i][1][2]] for i in (last_measurement_fuse_index+1):length(xs)])
      measurements_w = measurements[(last_measurement_fuse_index+1):end]
      measure_Σ = (σ_meas^2) * I(length(measurements_w));

      state_correction = stgpkf_correct(stgp_problem, state, measurement_pos, measurements_w, measure_Σ)
      state = stgpkf_predict(stgp_problem, state_correction)
      
      est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
      w_hat_new = reshape(est, length(EnvData.xs), length(EnvData.ys))

      push!(w_hat_ts, t)
      push!(w_hats, w_hat_new)

      qs = update_clarity_decay_from_trajectories(ergo_grid, ergo_q_maps[end], measurement_pos, Q_mat, EnvData)
      q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
      ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
      push!(ergo_q_maps, ergo_q_map)

      last_measurement_fuse_time = t
      last_measurement_fuse_index = length(measurements)
    end
    
    x = xs[end]
    b = bs[end]

    ys = [measure(EnvData, x[end]..., EnvData.ts[it], σ_meas)...]
    append!(measurements, ys)

    traj = vcat(xs...)
    M = w_hats[end]

    speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);
    push!(speeds, speed);
    
    u, q_target = controllers(t, x, M, w_rated_val, convex_polygon;
      ngpkf_grid=ngpkf_grid,
      ergo_grid=ergo_grid,
      ergo_q_map=ergo_q_maps[end],
      traj=traj,
      umax=speed,
      ΔT=ΔT,
    )
    
    u = norm(u) > 1e-8 ? normalize(u) * speed : zero(u)

    push!(q_target_maps, q_target)
    push!(us, u)

    u_step = us[end]
    new_xs = step(t, xs[end], u_step, ΔT/60)

    if any(isnan, new_xs)
      new_xs = xs[end]
    else
      new_xs = clamp_to_domain(new_xs, EnvData.xs, EnvData.ys)
    end

    push!(xs, new_xs)
    push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u_step), b, ΔT/60))
  end

  return SimResultWeightedSpeed(ts, xs, us, speeds, bs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)
end

function simulate_nongp_transect(ts, x0::XS, b0, controllers, soc_profile, w_rated_val, convex_polygon, stgp_problem;
  ngpkf_grid::G,
  EnvData,
  transect_pts,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0/60,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector, XS<:AbstractVector{X}, G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288;
  lat = 35.45;
  
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys) 
  ergo_grid = ErgoGrid(ngpkf_grid, (length(EnvData.xs), length(EnvData.ys)))

  w_hat_ts = [t0,]
  state = stgpkf_initialize(stgp_problem)
  est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
  w_hat = reshape(est, length(EnvData.xs), length(EnvData.ys))
  w_hats = [w_hat, ]

  qs = zeros(length(EnvData.xs) * length(EnvData.ys))
  q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
  Q_mat = zeros(length(EnvData.xs), length(EnvData.ys))
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  measurements = [measure(EnvData, x0[1]..., EnvData.ts[1], σ_meas)...]

  M = ones(Nx, Ny) * w_rated_val

  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0, soc_profile[1], error_sum, error);
  speeds = [speed];

  waypoint_idx = 1

  u0, q_target, waypoint_idx = controllers(t0, x0, M, w_rated_val, convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    transect_pts=transect_pts,
    waypoint_idx=waypoint_idx,
    umax=speed,
    ΔT=ΔT,
  )
  us = [u0,]

  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0/60, lat, norm(u0), b0, ΔT/60))
  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  @progress for (it, t) in enumerate(ts[1:(end-1)])
    if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT
      measurement_pos = vec([@SVector[xs[i][1][1], xs[i][1][2]] for i in (last_measurement_fuse_index+1):length(xs)])
      measurements_w = measurements[(last_measurement_fuse_index+1):end]
      measure_Σ = (σ_meas^2) * I(length(measurements_w));

      state_correction = stgpkf_correct(stgp_problem, state, measurement_pos, measurements_w, measure_Σ)
      state = stgpkf_predict(stgp_problem, state_correction)
      
      est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
      w_hat_new = reshape(est, length(EnvData.xs), length(EnvData.ys))

      push!(w_hat_ts, t)
      push!(w_hats, w_hat_new)

      qs = update_clarity_decay_from_trajectories(ergo_grid, ergo_q_maps[end], measurement_pos, Q_mat, EnvData)
      q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
      ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
      push!(ergo_q_maps, ergo_q_map)

      last_measurement_fuse_time = t
      last_measurement_fuse_index = length(measurements)
    end
    
    x = xs[end]
    b = bs[end]

    ys = [measure(EnvData, x[end]..., EnvData.ts[it], σ_meas)...]
    append!(measurements, ys)

    traj = vcat(xs...)
    M = w_hats[end]

    speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);
    push!(speeds, speed);
    
    u, q_target, waypoint_idx = controllers(t, x, M, w_rated_val, convex_polygon;
      ngpkf_grid=ngpkf_grid,
      ergo_grid=ergo_grid,
      ergo_q_map=ergo_q_maps[end],
      traj=traj,
      transect_pts=transect_pts,
      waypoint_idx=waypoint_idx,
      umax=speed,
      ΔT=ΔT,
    )
    
    push!(q_target_maps, q_target)
    push!(us, u)

    u_step = us[end]
    new_xs = step(t, xs[end], u_step, ΔT/60)
    new_xs = clamp_to_domain(new_xs, EnvData.xs, EnvData.ys)

    push!(xs, new_xs)
    push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u_step), b, ΔT/60))
  end

  return SimResultWeightedSpeed(ts, xs, us, speeds, bs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)
end

end