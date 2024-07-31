module JordanLakeDomain
export 𝐬 # Vector of vectors of x- and y-coordinates in units of km
export polygon # Boundary of region of interest in Jordan lake

using LazySets
# using Plots

const DIST_BT_LOCS = 0.2 # Horizontal/vertical distance between locations [km]
# Note: `DIST_BT_LOCS = 0.2` produces a `length(𝐬)` of 167

const polygon = Polygon([[0, 0],
    [2.1314, 0.9141],
    [1.8282, 1.2219],
    [0.9141, 0.9141],
    [0.6063, 1.5251],
    [0.9094, 3.0408],
    [0.9188, 3.3579],
    [1.21725, 3.9642],
    [0.9141, 4.2720],
    [1.2173, 5.4893],
    [1.8282, 6.1002],
    [1.5204, 6.4034],
    [0.9048, 6.5013],
    [0.9141, 5.4893],
    [-0.2332, 3.8710],
    [-0.3078, 2.7423],
    [-0.6063, 2.1360],
])
𝐬 = Vector{Vector{Float64}}() # points of polygon

let
    # plot_object = plot(polygon, aspect_ratio=:equal)
    # plot_object = xlims!(plot_object, -1, 2.5)

    # https://github.com/JuliaReach/LazySets.jl/blob/16371c1569458baee259b06b33fd5245ffed650d/src/Sets/Polygon/in.jl
    # Algorithm:
    # Choose an arbitrary ray through `x` and count whether the number of
    # intersections with edges is odd. We choose the ray that is vertical upward.
    # Vertical line segments are ignored (after checking whether `x` is a member).
    function ∈(x::AbstractVector, P::Polygon)
        @assert length(x) == dim(P) "a vector of length $(length(x)) is " *
                                    "incompatible with a $(dim(P))-dimensional set"

        N = promote_type(eltype(x), eltype(P))

        vlist = P.vertices
        if length(vlist) < 2
            if isempty(vlist)
                return false
            elseif length(vlist) == 1
                return @inbounds x == vlist[1]
            end
        end

        # vertical ray as a vertical line (compare y coordinates later)
        vline = Line2D([one(N), zero(N)], @inbounds x[1])

        p = @inbounds vlist[end]
        odd = false
        @inbounds for q in vlist
            if (p[1] <= x[1] && q[1] >= x[1]) || (q[1] <= x[1] && p[1] >= x[1])
                # line segment pq intersects vertical line through x
                if p[1] == q[1]
                    # vertical line segment
                    if (p[2] <= x[2] && q[2] >= x[2]) || (q[2] <= x[2] && p[2] >= x[2])
                        # x is on the line segment
                        return true
                    end
                else
                    # non-vertical line segment -> intersect (Line2D intersection is used)
                    line2 = Line2D(p, q)
                    y = element(intersection(line2, vline))
                    # compare y coordinate
                    if y[2] >= x[2]
                        if y == x
                            # x is on line segment
                            return true
                        end
                        odd = !odd
                    end
                end
            end

            p = q
        end
        return odd
    end

    bounding_box = overapproximate(polygon)
    c = bounding_box.center
    r = bounding_box.radius
    # plot_object = plot!(plot_object, bounding_box)

    mesh_x = range(c[1] - r[1], c[1] + r[1], step=DIST_BT_LOCS)
    mesh_y = range(c[2] - r[2], c[2] + r[2], step=DIST_BT_LOCS)

    points_of_bounding_box = [[x, y] for y in mesh_y for x in mesh_x]

    for p in points_of_bounding_box
        if p ∈ polygon
            push!(𝐬, p)
            # plot_object = scatter!(plot_object, [point[1]], [point[2]],
            #     color=:green, ms=2, legend=false)
        end
    end
    # display(plot_object)
end

end
