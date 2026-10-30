"""Scalar sampled field with axes x/y in metres and time in seconds.

values[i,j,k] corresponds to xs[i], ys[j], times[k]. Uses linear interpolation
and rejects extrapolation. A singleton axis is valid only at that coordinate.
Loaders for CSV/NetCDF can construct this type after converting coordinates/units.
"""
struct GriddedField <: AbstractField
    xs::Vector{Float64}
    ys::Vector{Float64}
    times::Vector{Float64}
    values::Array{Float64,3}
    function GriddedField(xs, ys, times, values)
        axes = map(x -> Float64.(collect(x)), (xs, ys, times))
        for axis in axes
            !isempty(axis) && all(isfinite, axis) && all(diff(axis) .> 0) ||
                throw(ArgumentError("field axes must be finite, nonempty, and strictly increasing"))
        end
        size(values) == Tuple(length.(axes)) || throw(DimensionMismatch("field shape must be (nx, ny, nt)"))
        all(isfinite, values) || throw(ArgumentError("field values must be finite"))
        new(axes..., copy(Float64.(values)))
    end
end
function bracket(axis, value)
    isfinite(value) && first(axis) <= value <= last(axis) ||
        throw(DomainError(value, "query is outside sampled field bounds"))
    length(axis) == 1 && return (1, 1, 0.0)
    i = min(searchsortedlast(axis, value), length(axis) - 1)
    (i, i + 1, (value - axis[i]) / (axis[i+1] - axis[i]))
end
function field_value(field::GriddedField, position, time)
    p = point2(position)
    ix, jx, wx = bracket(field.xs, p[1])
    iy, jy, wy = bracket(field.ys, p[2])
    it, jt, wt = bracket(field.times, time)
    v = 0.0
    for (i,a) in ((ix,1-wx),(jx,wx)), (j,b) in ((iy,1-wy),(jy,wy)), (k,c) in ((it,1-wt),(jt,wt))
        v += a*b*c*field.values[i,j,k]
    end
    v
end
