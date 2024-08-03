module JordanLakeDomain
export 𝐬 # Vector of vectors of x- and y-coordinates in units of km
export polygon # Boundary of region of interest in Jordan lake

using LazySets

const DIST_BT_LOCS = 0.15 # Horizontal/vertical distance between locations [km]
# Note: `DIST_BT_LOCS = 0.2` produces a `length(𝐬)` of 114
#       `DIST_BT_LOCS = 0.15` produces a `length(𝐬)` of 203
#       `DIST_BT_LOCS = 0.1` produces a `length(𝐬)` of 458

const polygon = VPolygon([[0, 0],
    [0.3362, 0],
    [0.7925, 4.3707],
    [0.9366, 6.3399],
    [-0.1441, 4.7069],
    [-0.3602, 1.3929],
])
𝐬 = Vector{Vector{Float64}}() # points of polygon

let
    bounding_box = overapproximate(polygon)
    c = bounding_box.center
    r = bounding_box.radius

    mesh_x = range(c[1] - r[1], c[1] + r[1], step=DIST_BT_LOCS)
    mesh_y = range(c[2] - r[2], c[2] + r[2], step=DIST_BT_LOCS)

    points_of_bounding_box = [[x, y] for y in mesh_y for x in mesh_x]

    for p in points_of_bounding_box
        if p ∈ polygon
            push!(𝐬, p)
        end
    end
end

end
