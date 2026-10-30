"""A scalar synthetic field f(position_m, time_s), independent of any estimator."""
struct AnalyticField{F} <: AbstractField
    f::F
end
function field_value(field::AnalyticField, position, time)
    value = Float64(field.f(point2(position), time))
    isfinite(value) || throw(ArgumentError("field returned a nonfinite value"))
    value
end
