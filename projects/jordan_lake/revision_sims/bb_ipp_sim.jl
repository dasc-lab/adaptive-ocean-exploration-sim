using Plots
using LinearAlgebra

# ==============================================================================
# 1. Multi-Pocket Spatiotemporal Target Field (Non-GP Dynamic Environment)
# ==============================================================================
"""
    target_field(x, y, t)

Spatiotemporally varying environment engineered with dynamic local reward pockets.
Myopic lookahead planners (like BB-IPP with horizon H <= 4) get drawn into local
attractor wells and trapped near local maxima, whereas global ergodic exploration
diffuses coverage across all dynamic pockets.
"""
function target_field(x::Float64, y::Float64, t::Float64=0.0)
    # Pocket 1: Local attractor well (trap near initial starting region)
    cx1, cy1 = -2.2 + 0.15 * sin(0.12 * t), -2.0 + 0.15 * cos(0.12 * t)
    r1 = 1.25 * exp(-((x - cx1)^2 + (y - cy1)^2) / 0.45)

    # Pocket 2: Secondary local trap pocket
    cx2, cy2 = -1.2 + 0.1 * cos(0.1 * t), 1.8 + 0.1 * sin(0.1 * t)
    r2 = 0.95 * exp(-((x - cx2)^2 + (y - cy2)^2) / 0.55)

    # Pocket 3: Distal high-value target (separated by a low-reward valley barrier)
    cx3, cy3 = 2.5 + 0.7 * cos(0.08 * t), 2.2 + 0.7 * sin(0.08 * t)
    r3 = 1.60 * exp(-((x - cx3)^2 + (y - cy3)^2) / 1.4)

    # Pocket 4: Oscillating saddle pocket
    cx4, cy4 = 1.2, -1.8
    r4 = 0.8 * exp(-((x - cx4)^2 + (y - cy4)^2) / 0.75) * (1.0 + 0.25 * sin(0.2 * t))

    return r1 + r2 + r3 + r4
end

# ==============================================================================
# 2. Motion Primitives
# ==============================================================================
struct MotionPrimitive
    dtheta::Float64  # Heading change
    dist::Float64    # Step length (v * dt)
end

function get_motion_primitives(speed=0.5, dt=1.0, M=5)
    dthetas = range(-pi/3, pi/3, length=M)
    return [MotionPrimitive(dt_angle, speed * dt) for dt_angle in dthetas]
end

# ==============================================================================
# 3. Branch & Bound Informative Path Planning (BB-IPP)
# ==============================================================================
mutable struct BBState
    gamma_star::Float64      # Incumbent best total reward (γ*)
    z_star::Vector{Int}      # Incumbent best primitive sequence (z*)
end

function bb_recursion!(x::Vector{Float64}, z_parent::Vector{Int}, gamma_parent::Float64, 
                         j::Int, H::Int, L_UB::Float64, primitives::Vector{MotionPrimitive}, 
                         state::BBState, heading::Float64, t::Float64, tree_lines::Vector)
    
    # Upper bound approximation over remaining horizon
    if j < H
        gamma_max = gamma_parent + (H - j) * L_UB
    else
        gamma_max = -Inf
    end

    # Branching & pruning condition
    if gamma_max > state.gamma_star
        children = []
        
        for (i, prim) in enumerate(primitives)
            new_heading = heading + prim.dtheta
            new_x = x + [prim.dist * cos(new_heading), prim.dist * sin(new_heading)]
            
            reward_i = target_field(new_x[1], new_x[2], t + j)
            push!(tree_lines, (x, new_x))
            push!(children, (index=i, pos=new_x, heading=new_heading, reward=reward_i))
        end

        # Heuristic node ordering (descending reward)
        sort!(children, by = c -> c.reward, rev=true)

        for child in children
            gamma_child = gamma_parent + child.reward
            z_child = vcat(z_parent, child.index)

            if gamma_child > state.gamma_star
                state.gamma_star = gamma_child
                state.z_star = z_child
            end

            bb_recursion!(child.pos, z_child, gamma_child, j + 1, H, L_UB, 
                           primitives, state, child.heading, t, tree_lines)
        end
    end
end

function path_planning_bb(x_start::Vector{Float64}, heading::Float64, t_curr::Float64, 
                          H::Int, primitives::Vector{MotionPrimitive}, L_UB::Float64)
    state = BBState(-Inf, Int[])
    tree_lines = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
    
    bb_recursion!(x_start, Int[], 0.0, 0, H, L_UB, primitives, state, heading, t_curr, tree_lines)
    return state.z_star, state.gamma_star, tree_lines
end

function reconstruct_trajectory(x_start, heading, z_indices, primitives)
    pts = [copy(x_start)]
    curr_x, curr_h = copy(x_start), heading
    for idx in z_indices
        prim = primitives[idx]
        curr_h += prim.dtheta
        curr_x += [prim.dist * cos(curr_h), prim.dist * sin(curr_h)]
        push!(pts, copy(curr_x))
    end
    return pts
end

# ==============================================================================
# 4. Simulation Execution & GIF/MP4 Rendering
# ==============================================================================
function run_bb_ipp_animation()
    primitives = get_motion_primitives(0.5, 1.0, 5) # M = 5 primitives
    H = 4                                            # Lookahead horizon H = 4
    L_UB = 1.6                                       # Upper bound constant
    T_sim = 40                                       # Simulation duration
    frame_size = (1920, 1080)
    
    x_curr = [-2.5, -2.5]
    heading_curr = pi/4
    executed_path = [copy(x_curr)]
    
    xs = range(-5, 5, length=80)
    ys = range(-5, 5, length=80)

    default(size=frame_size, titlefontsize=28, guidefontsize=22,
            tickfontsize=16, legendfontsize=18)
    
    println("Generating BB-IPP simulation animation with multi-pocket environment...")
    
    anim = @animate for t in 1:T_sim
        t_time = Float64(t)
        
        z_star, gamma_star, tree_lines = path_planning_bb(x_curr, heading_curr, t_time, H, primitives, L_UB)
        planned_pts = reconstruct_trajectory(x_curr, heading_curr, z_star, primitives)
        
        Z = [target_field(x, y, t_time * 0.1) for y in ys, x in xs]
        
        plt = heatmap(xs, ys, Z, c=:viridis, clims=(0.0, 1.8),
                      title="BB-IPP Local Pocket Trap Simulation | Step t = $t",
                      xlabel="x [m]", ylabel="y [m]", aspect_ratio=:equal, 
                      xlims=(-5,5), ylims=(-5,5), legend=:topleft)
        
        # Render expanded/pruned search tree branches
        for (p1, p2) in tree_lines
            plot!(plt, [p1[1], p2[1]], [p1[2], p2[2]], color=:white, alpha=0.12, label="")
        end
        
        # Render planned and executed trajectories
        pxs = [p[1] for p in planned_pts]
        pys = [p[2] for p in planned_pts]
        plot!(plt, pxs, pys, color=:cyan, linewidth=2.5, marker=:circle, markersize=3, label="BB-IPP Plan (z*)")
        
        exs = [p[1] for p in executed_path]
        eys = [p[2] for p in executed_path]
        plot!(plt, exs, eys, color=:red, linewidth=2.5, label="Executed Path")
        scatter!(plt, [x_curr[1]], [x_curr[2]], color=:red, markersize=6, label="Vehicle")
        
        # Receding-Horizon step execution
        if !isempty(z_star)
            first_prim = primitives[z_star[1]]
            heading_curr += first_prim.dtheta
            x_curr += [first_prim.dist * cos(heading_curr), first_prim.dist * sin(heading_curr)]
            push!(executed_path, copy(x_curr))
        end
    end
    
    mp4(anim, "bb_ipp_pockets_demo.mp4", fps=4)
    gif(anim, "bb_ipp_pockets_demo.gif", fps=4)
    println("Animation saved: bb_ipp_pockets_demo.mp4 and bb_ipp_pockets_demo.gif")
end

run_bb_ipp_animation()