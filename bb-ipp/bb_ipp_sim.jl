using Plots
using LinearAlgebra

# ==============================================================================
# 1. Target Clarity Field (Scalar Wind Resource / Clarity Map)
# ==============================================================================
function target_field(x::Float64, y::Float64, t::Float64=0.0)
    # Moving dynamic target peaks near rated wind speeds
    cx1, cy1 = 2.0 + 1.5 * cos(0.1 * t), 2.0 + 1.5 * sin(0.1 * t)
    cx2, cy2 = -2.0, -1.0
    
    r1 = exp(-((x - cx1)^2 + (y - cy1)^2) / 3.0)
    r2 = 0.8 * exp(-((x - cx2)^2 + (y - cy2)^2) / 4.0)
    return r1 + r2
end

# ==============================================================================
# 2. Motion Primitives
# ==============================================================================
struct MotionPrimitive
    dtheta::Float64  # Heading change
    dist::Float64    # Step length (v * dt)
end

function get_motion_primitives(speed=0.5, dt=1.0, M=5)
    dthetas = range(-pi/3, pi/3, length=M) # Fan of M motion primitives
    return [MotionPrimitive(dt_angle, speed * dt) for dt_angle in dthetas]
end

# ==============================================================================
# 3. Algorithm 2: Branch & Bound Informative Path Planning (BB-IPP)
# ==============================================================================
mutable struct BBState
    gamma_star::Float64      # Incumbent best total reward (γ*)
    z_star::Vector{Int}      # Incumbent best primitive sequence (z*)
end

"""
Line 8: Recursive Branch & Bound Tree Search
"""
function bb_recursion!(x::Vector{Float64}, z_parent::Vector{Int}, gamma_parent::Float64, 
                         j::Int, H::Int, L_UB::Float64, primitives::Vector{MotionPrimitive}, 
                         state::BBState, heading::Float64, t::Float64, tree_lines::Vector)
    
    # Lines 9-14: Upper Bound Approximation over remaining horizon
    if j < H
        gamma_max = gamma_parent + (H - j) * L_UB
    else
        gamma_max = -Inf
    end

    # Line 15: Branching & Pruning Condition (γ_max > γ*)
    if gamma_max > state.gamma_star
        children = []
        
        # Lines 16-18: Sample M primitives & evaluate rewards
        for (i, prim) in enumerate(primitives)
            new_heading = heading + prim.dtheta
            new_x = x + [prim.dist * cos(new_heading), prim.dist * sin(new_heading)]
            
            # Predict sample reward γ_i
            reward_i = target_field(new_x[1], new_x[2], t + j)
            
            # Record tree edge for visualization
            push!(tree_lines, (x, new_x))
            
            push!(children, (index=i, pos=new_x, heading=new_heading, reward=reward_i))
        end

        # Line 19: Sort Z by decreasing reward γ_i (Heuristic node ordering)
        sort!(children, by = c -> c.reward, rev=true)

        # Lines 20-26: Expand children
        for child in children
            gamma_child = gamma_parent + child.reward
            z_child = vcat(z_parent, child.index)

            # Lines 22-24: Update best incumbent solution
            if gamma_child > state.gamma_star
                state.gamma_star = gamma_child
                state.z_star = z_child
            end

            # Line 25: Recurse to next depth (j + 1)
            bb_recursion!(child.pos, z_child, gamma_child, j + 1, H, L_UB, 
                        primitives, state, child.heading, t, tree_lines)
        end
    end
end

"""
Line 1: Main BB-IPP Procedure
"""
function path_planning_bb(x_start::Vector{Float64}, heading::Float64, t_curr::Float64, 
                          H::Int, primitives::Vector{MotionPrimitive}, L_UB::Float64)
    # Line 2: Initialize incumbent best reward γ* = 0 (or -Inf)
    state = BBState(-Inf, Int[])
    tree_lines = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
    
    # Line 6: Execute recursive search
    bb_recursion!(x_start, Int[], 0.0, 0, H, L_UB, primitives, state, heading, t_curr, tree_lines)
    
    return state.z_star, state.gamma_star, tree_lines
end

# Reconstruct 2D trajectory points from primitive indices
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
# 4. Simulation Execution & GIF Rendering
# ==============================================================================
function run_bb_ipp_animation()
    primitives = get_motion_primitives(0.5, 1.0, 5) # M = 5 primitives
    H = 4             # Lookahead horizon H = 4
    L_UB = 1.0        # Upper bound constant (max single-step reward)
    T_sim = 35        # Simulation duration
    frame_size = (1920, 1080)  # 16:9 1080p output
    title_font = 28
    guide_font = 22
    tick_font = 16
    legend_font = 18
    
    # Vehicle initial state
    x_curr = [-3.0, -3.0]
    heading_curr = pi/4
    executed_path = [copy(x_curr)]
    
    # Grid domain for background visualization
    xs = range(-5, 5, length=80)
    ys = range(-5, 5, length=80)

        default(size=frame_size, titlefontsize=title_font, guidefontsize=guide_font,
            tickfontsize=tick_font, legendfontsize=legend_font)
    
    println("Generating BB-IPP simulation animation...")
    
    anim = @animate for t in 1:T_sim
        t_time = Float64(t)
        
        # 1. Run BB-IPP Path Planner (Algorithm 2)
        z_star, gamma_star, tree_lines = path_planning_bb(x_curr, heading_curr, t_time, H, primitives, L_UB)
        
        # 2. Reconstruct planned path
        planned_pts = reconstruct_trajectory(x_curr, heading_curr, z_star, primitives)
        
        # 3. Render Field Heatmap
        Z = [target_field(x, y, t_time*0.1) for y in ys, x in xs]
        
        plt = heatmap(xs, ys, Z, c=:viridis, clims=(0.0, 1.2),
                  size=frame_size,
                      titlefontsize=title_font,
                      guidefontsize=guide_font,
                      tickfontsize=tick_font,
                      legendfontsize=legend_font,
                      title="Branch & Bound IPP (BB-IPP) | Step t = $t",
                      xlabel="x [m]", ylabel="y [m]", aspect_ratio=:equal, 
                      xlims=(-5,5), ylims=(-5,5), legend=:topleft)
        
        # Render expanded/pruned search tree branches (Faint white lines)
        for (p1, p2) in tree_lines
            plot!(plt, [p1[1], p2[1]], [p1[2], p2[2]], color=:white, alpha=0.12, label="")
        end
        
        # Render optimal lookahead plan z* (Cyan line)
        pxs = [p[1] for p in planned_pts]
        pys = [p[2] for p in planned_pts]
        plot!(plt, pxs, pys, color=:cyan, linewidth=2.5, marker=:circle, markersize=3, label="BB-IPP Plan (z*)")
        
        # Render past executed path (Red line)
        exs = [p[1] for p in executed_path]
        eys = [p[2] for p in executed_path]
        plot!(plt, exs, eys, color=:red, linewidth=2.5, label="Executed Path")
        scatter!(plt, [x_curr[1]], [x_curr[2]], color=:red, markersize=6, label="Vehicle")
        
        # 4. Receding-Horizon Step: Execute 1st action from z_star
        if !isempty(z_star)
            first_prim = primitives[z_star[1]]
            heading_curr += first_prim.dtheta
            x_curr += [first_prim.dist * cos(heading_curr), first_prim.dist * sin(heading_curr)]
            push!(executed_path, copy(x_curr))
        end
    end
    
    # Save output as high-resolution video and GIF using the same rendered canvas.
    save_path_mp4 = "bb_ipp_demo_1080p.mp4"
    save_path_gif = "bb_ipp_demo_1080p.gif"
    mp4(anim, save_path_mp4, fps=3)
    gif(anim, save_path_gif, fps=3)
    println("Animation saved successfully to $save_path_mp4 and $save_path_gif")
end

run_bb_ipp_animation()
