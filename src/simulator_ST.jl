module SimulatorST

using LinearAlgebra, StaticArrays, Interpolations
using ProgressLogging
using ..SyntheticData, ..NGPKF, ..ErgodicController, ..KF, ..SoCController, ..Variograms, ..STGPKF, ..JordanLakeDomain


# MATERN SPATIAL LENGTH SCALE = 1.0 km
# MATERN TEMPORAL LENGTH SCALE = 0.2 * 60 * 24 = 4.8 hrs

struct MeasurementSpatial{T, P, F}
  t::T
  p::P
  y::F
end

"""
  take_measurement(t, p, data; σ_meas = 0, Q_meas = σ_meas*I)

returns a SVector of the [wx, wy] at time t, and pos p by querying the data 
"""

function measure(t, p::SV, data::EDST; σ_meas=0, Q_meas=σ_meas * I) where {EDST<:EnvDataST,SV<:SVector{2}}
  y = data(p..., t) + (σ_meas * randn(1))[1]
  return MeasurementSpatial(t, p, y)
end

function measure(t, ps::VSV, data::EDST; σ_meas=0, Q_meas = σ_meas * I) where {EDST<:EnvDataST,SV<:SVector{2}, VSV <: AbstractVector{SV}}
  return [measure(t, p, data; Q_meas=Q_meas) for p in ps]

end


function step(t, x::X, u::U, ΔT) where {X<:SVector,U<:SVector}

  A = I(2)
  B = ΔT * I(2)

  return A * x + B * u
end


function step(t, xs::XS, us::US, ΔT) where {X<:SVector,U<:SVector,XS<:AbstractVector{X},US<:AbstractVector{U}}

  length(xs) == length(us) || throw(DimensionMismatch())

  N = length(xs)

  return [step(t, xs[i], us[i], ΔT) for i = 1:N]

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

function ErgoGrid(ngpkf_grid::G) where {G<:NGPKF.NGPKFGrid}
  origin = (ngpkf_grid.xs[1], ngpkf_grid.ys[1])
  dxs = (Base.step(ngpkf_grid.xs), Base.step(ngpkf_grid.ys))
  Ls = (maximum(ngpkf_grid.xs) - minimum(ngpkf_grid.xs), maximum(ngpkf_grid.ys) - minimum(ngpkf_grid.ys))

  ergo_grid = ErgodicController.Grid(origin, dxs, Ls)

  return ergo_grid
end

# L = dx * (N-1)

function ErgoGrid(ngpkf_grid::G, Ns) where {G<:NGPKF.NGPKFGrid}

  origin = (ngpkf_grid.xs[1], ngpkf_grid.ys[1])
  Ls = (maximum(ngpkf_grid.xs) - minimum(ngpkf_grid.xs), maximum(ngpkf_grid.ys) - minimum(ngpkf_grid.ys))
  dxs = Ls ./ (Ns .- 1)

  # Creates an ergo_grid with origin, spacing which accounts for NGPKF grid and then the plan for direction cosine transform
  ergo_grid = ErgodicController.Grid(origin, dxs, Ls)

  return ergo_grid

end

function ngpkf_to_ergo(ngpkf_grid::G1, ergo_grid::G2, clarity_map) where {G1<:NGPKF.NGPKFGrid,G2<:ErgodicController.Grid}

  Ns = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  if Ns == ergo_grid.N
    return 1.0 * clarity_map
  end

  # interpolate the data
  itp = linear_interpolation((ngpkf_grid.xs, ngpkf_grid.ys), clarity_map, extrapolation_bc=Line())
  ergo_map = itp(ErgodicController.xs(ergo_grid), ErgodicController.ys(ergo_grid))

  return ergo_map

end

# Simulate pre-recorded trajectory 


# Simulate Spatial only 
function simulate_spatial(ts, x0::XS, controllers;
    ngpkf_grid::G,
    EnvDataST,
    σ_meas=0, 
    σ_process=0,
    Q_process = σ_process^2 * I,
    fuse_measurements_every_ΔT=5.0,
    recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}
  
    
    # extract info from arguments
    t0 = ts[1]
    xs = [x0,]
    N_robots = length(x0)
    ΔT = Base.step(ts)
  
    Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
    ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))
  
    # setup map states
    w_hat_ts = [t0,]
  
    w_hat = NGPKF.initialize(ngpkf_grid)
    w_hats = [w_hat,]

    # check the covariances
    # print(NGPKF.KF.Σ(w_hat))
    # @assert false

    # clarity map
    q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
    ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
    ergo_q_maps = [ergo_q_map,]
  
    # get a measurement
    # ys = [measure(t0, x0[i], EnvDataST; σ_meas=σ_meas) for i = 1:N_robots]
    println("going into measurements")
    measurements = [measure(t0, x0, EnvDataST; σ_meas = σ_meas)...]  
    println("going out of measurements")
    # decide the control input for the first step
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
  
    last_control_update_time = t0
    # try
      @progress for (it, t) in enumerate(ts[1:(end-1)])
        
        x = xs[end]


        # make a measurement from each robot
        ys = measure(t, x, EnvDataST; σ_meas=σ_meas)
        append!(measurements, ys)
        # check if we need to fuse measurements
        if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT
  
          # # collect all the locations we have made measurements
          # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
          # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)
  
          # # extract x and y components of the measurements
          # measurements_wx = [w[1] for w in measurement_w]
          # measurements_wy = [w[2] for w in measurement_w]
  
          # run NGPKF
          w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

          # grab the data again

          new_measurements = measurements[(last_measurement_fuse_index+1):end]
          measurement_pos = [m.p for m in new_measurements]
          measurements_w = [m.y for m in new_measurements]

          # run the fusion
          w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
          
          # save the new maps
          push!(w_hat_ts, t)
          push!(w_hats, w_hat_new)
  
          # update the clarity map
          q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
          ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
          push!(ergo_q_maps, ergo_q_map)
  
          last_measurement_fuse_time = t
          last_measurement_fuse_index = length(measurements)
        end
  
  
        # if (t - last_control_update_time >= recompute_controller_every_ΔT)
        #   # chose a control action
  
          traj = vcat(xs...)

          u = controllers(t, x;
            ngpkf_grid=ngpkf_grid,
            ergo_grid=ergo_grid,
            ergo_q_map=ergo_q_maps[end], # current clarity
            traj=traj,  # list of all points visited by all agents
            ΔT=ΔT,
          )
          
        #   println("u : $(u)")
          push!(us, u)
  
        #   last_control_update_time = t
  
        # end
  
  
  
        # update 
        u = us[end] # use the last control input
        new_xs = step(t, xs[end], u, ΔT)
        push!(xs, new_xs)
  
      end
  
    # catch e
      # println(e)
    # end
  
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
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  N_robots = length(x0)
  ΔT = Base.step(ts)
  
  Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  wx_hat = NGPKF.initialize(ngpkf_grid)
  wx_hats = [wx_hat,]

  wy_hat = NGPKF.initialize(ngpkf_grid)
  wy_hats = [wy_hat,]

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, wx_hat, wy_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataST; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvDataST; σ_meas = σ_meas)...]  

  # decide the control input for the first step
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

  last_control_update_time = t0

  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])

      x = xs[end]
 
      # make a measurement from each robot
      ys = measure(t, x, EnvDataST; σ_meas=σ_meas)
      append!(measurements, ys)

      

      # check if we need to fuse measurements
      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # # collect all the locations we have made measurements
        # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
        # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)

        # # extract x and y components of the measurements
        # measurements_wx = [w[1] for w in measurement_w]
        # measurements_wy = [w[2] for w in measurement_w]

        # run NGPKF
        wx_hat = NGPKF.predict(ngpkf_grid, wx_hats[end]; Q_process=Q_process)
        wy_hat = NGPKF.predict(ngpkf_grid, wy_hats[end]; Q_process=Q_process)
        

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_wx = [m.y[1] for m in new_measurements]
        measurements_wy = [m.y[2] for m in new_measurements]

        # run the fusion
        wx_hat_new = NGPKF.correct(ngpkf_grid, wx_hat, measurement_pos, measurements_wx; σ_meas=σ_meas)
        wy_hat_new = NGPKF.correct(ngpkf_grid, wy_hat, measurement_pos, measurements_wy; σ_meas=σ_meas)

        # save the new maps
        push!(w_hat_ts, t)
        push!(wx_hats, wx_hat_new)
        push!(wy_hats, wy_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, wx_hat_new, wy_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        u = controllers(t, x;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          ΔT=ΔT,
        )
        
        push!(us, u)

      #   last_control_update_time = t

      # end
      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)
      push!(xs, new_xs)

    end

  # catch e
    # println(e)
  # end

  return SimResult(ts, xs, us, measurements, w_hat_ts, wx_hats, wy_hats, ergo_q_maps)

end

# Spatiotemporal Jordan Lake Simulation with known hyperparameters
function simulate_known_param(ts, x0::XS, b0, controllers, soc_profile, w_rated_val, convex_polygon, params, estimator;
  ngpkf_grid::G,
  EnvData,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0/60,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288; # corresponds to October 15th
  lat = 35.45; # degrees
  
  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0;
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Nx, Ny= length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  # TODO: Replace with STGPKF.initialize()
  w_hat = NGPKF.initialize(ngpkf_grid)
  # STGPKF.initialize!(estimator, params)
  # w_hat = reshape(estimator.𝐱ʰᵃᵗⱼₗᵢ[:, 1], Nx, Ny)
  w_hats = [w_hat,]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvData; σ_meas = σ_meas)...]  

  # Initiatize the estimate to be equal to the rated value
  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  M = ones(Nx, Ny)
  M *= w_rated_val

  # Call the real-time speed controller
  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0 ,soc_profile[1], error_sum, error);

  speeds = [speed];

  # decide the control input for the first step
  u0, q_target = controllers(t0, x0, M, w_rated_val,convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    umax=speed,
    ΔT=ΔT,
  )
  us = [u0,]

  # update ASV state of charge
  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0, lat, norm(u0), b0, ΔT))

  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0
  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])
      
      x = xs[end]
      b = bs[end]

      # make a measurement from each robot
      ys = measure(t, x, EnvData; σ_meas=σ_meas)
      append!(measurements, ys)
      

      # STGPKF.update_and_predict!(estimator, params, ys.y, ys.p)

      # w_hat = estimator.𝐱ʰᵃᵗⱼₗᵢ[:, 1]
      # w_hat = reshape(w_hat, Nx, Ny)
       # run NGPKF
      #  w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

       # grab the data again
      #  new_measurements = measurements[(last_measurement_fuse_index+1):end]
      #  measurement_pos = [m.p for m in new_measurements]
      #  measurements_w = [m.y for m in new_measurements]

       # run the fusion
      #  w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
       
      #  STGPKF.update_and_predict!(estimator_1, key_parameters_1, yᵢ, 𝐬ᵢᵗⁱˡᵈᵉ)

       # save the new maps
      #  push!(w_hat_ts, t)
      #  push!(w_hats, w_hat_new)

       # update the clarity map
      #  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
      #  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
      #  push!(ergo_q_maps, ergo_q_map)

      #  last_measurement_fuse_time = t
      #  last_measurement_fuse_index = length(measurements)


      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # # collect all the locations we have made measurements
        # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
        # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)

        # # extract x and y components of the measurements
        # measurements_wx = [w[1] for w in measurement_w]
        # measurements_wy = [w[2] for w in measurement_w]

        # run STGPKF
        # STGPKF.update_and_predict!(estimator, params, ys[end], 𝐬ᵢᵗⁱˡᵈᵉ)

        # run NGPKF
        w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_w = [m.y for m in new_measurements]

        # run the fusion
        w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # STGPKF.update_and_predict!(estimator_1, key_parameters_1, yᵢ, 𝐬ᵢᵗⁱˡᵈᵉ)

        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        M = reshape(KF.μ(w_hats[end]), Nx, Ny)

        speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);
        push!(speeds, speed);
        
        u, q_target = controllers(t, x, M, w_rated_val,convex_polygon;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          umax=speed,
          ΔT=ΔT,
        )
        
        push!(q_target_maps, q_target)
        # Debug 01
        # if isnan(x[1])
        #   println("current state: $(x)")
        # end


        push!(us, u)

      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)

      # if isnan(new_xs[1])
      #   println("New state: $(new_xs)")
      # end
      push!(xs, new_xs)
      push!(bs, SoCController.batterymodel!(boat, dayOfYear, t, lat, norm(u0), b, ΔT))

    end

  # catch e
    # println(e)
  # end

  return SimResultWeightedSpeed(ts, xs, us, speeds, bs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)

end

# Spatiotemporal Jordan Lake Simulation with estimated parameters
function simulate_param_est(ts, x0::XS, b0, controllers, soc_profile, w_rated_val, convex_polygon, params, estimator;
  ngpkf_grid::G,
  EnvData,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0/60,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288; # corresponds to October 15th
  lat = 35.45; # degrees

  # store hyperparameter estimates
  σs = [1.0, ]
  λxs = [1.0, ]
  λts = [1.0, ]

  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0;
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Nx, Ny= length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  # TODO: Replace with STGPKF.initialize()
  w_hat = NGPKF.initialize(ngpkf_grid)
  # STGPKF.initialize!(estimator, params)
  # w_hat = reshape(estimator.𝐱ʰᵃᵗⱼₗᵢ[:, 1], Nx, Ny)
  w_hats = [w_hat,]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvData; σ_meas = σ_meas)...]  

  # Initiatize the estimate to be equal to the rated value
  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  M = ones(Nx, Ny)
  M *= w_rated_val

  # Call the real-time speed controller
  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0 ,soc_profile[1], error_sum, error);

  speeds = [speed];

  # decide the control input for the first step
  u0, q_target = controllers(t0, x0, M, w_rated_val,convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    umax=speed,
    ΔT=ΔT,
  )
  us = [u0,]

  # update ASV state of charge
  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0, lat, norm(u0), b0, ΔT))

  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0
  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])
      
      x = xs[end]
      b = bs[end]

      # make a measurement from each robot
      ys = measure(t, x, EnvData; σ_meas=σ_meas)
      append!(measurements, ys)
      

      # STGPKF.update_and_predict!(estimator, params, ys.y, ys.p)

      # w_hat = estimator.𝐱ʰᵃᵗⱼₗᵢ[:, 1]
      # w_hat = reshape(w_hat, Nx, Ny)
       # run NGPKF
      #  w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

       # grab the data again
      #  new_measurements = measurements[(last_measurement_fuse_index+1):end]
      #  measurement_pos = [m.p for m in new_measurements]
      #  measurements_w = [m.y for m in new_measurements]

       # run the fusion
      #  w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
       
      #  STGPKF.update_and_predict!(estimator_1, key_parameters_1, yᵢ, 𝐬ᵢᵗⁱˡᵈᵉ)

       # save the new maps
      #  push!(w_hat_ts, t)
      #  push!(w_hats, w_hat_new)

       # update the clarity map
      #  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
      #  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
      #  push!(ergo_q_maps, ergo_q_map)

      #  last_measurement_fuse_time = t
      #  last_measurement_fuse_index = length(measurements)


      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # # collect all the locations we have made measurements
        # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
        # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)

        # # extract x and y components of the measurements
        # measurements_wx = [w[1] for w in measurement_w]
        # measurements_wy = [w[2] for w in measurement_w]

        # run STGPKF
        # STGPKF.update_and_predict!(estimator, params, ys[end], 𝐬ᵢᵗⁱˡᵈᵉ)

         # Fit new hyperparameters
         σ, λx, λt = Variograms.hp_fit(measurements)
         push!(σs, σ)
         push!(λxs, λx)
         push!(λts, λt)

         # Update KF
        res_factor = 0.1 #l_spatial / sqrt(2.0)

        kern = NGPKF.MaternKernel(σ, 1/λx)

        ngp_grid_x = range(extrema(EnvData.X)..., step= res_factor )
        ngp_grid_y = range(extrema(EnvData.Y)..., step= res_factor )

        ngpkf_grid = NGPKF.NGPKFGrid(ngp_grid_x, ngp_grid_y, kern)

        # run NGPKF
        w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_w = [m.y for m in new_measurements]

        # run the fusion
        w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # STGPKF.update_and_predict!(estimator_1, key_parameters_1, yᵢ, 𝐬ᵢᵗⁱˡᵈᵉ)

        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        M = reshape(KF.μ(w_hats[end]), Nx, Ny)

        speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);
        push!(speeds, speed);
        
        u, q_target = controllers(t, x, M, w_rated_val,convex_polygon;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          umax=speed,
          ΔT=ΔT,
        )
        
        push!(q_target_maps, q_target)
        # Debug 01
        # if isnan(x[1])
        #   println("current state: $(x)")
        # end


        push!(us, u)

      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)

      # if isnan(new_xs[1])
      #   println("New state: $(new_xs)")
      # end
      push!(xs, new_xs)
      push!(bs, SoCController.batterymodel!(boat, dayOfYear, t, lat, norm(u0), b, ΔT))

    end

  # catch e
    # println(e)
  # end
  
  return SimResultWeightedSpeedParams(ts, xs, us, speeds, bs, σs, λxs, λts, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)

end

end