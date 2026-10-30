"""Export lightweight trajectory data for plotting, outside the simulation loop.

Creates parent directories. Does not write datasets or outputs automatically.
"""
function write_trajectory_csv(path::AbstractString, result::SimulationResult)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "time_s,x_m,y_m,battery_wh")
        for (time,state) in zip(result.times, result.states)
            println(io, time, ',', state.position[1], ',', state.position[2], ',', state.battery_wh)
        end
    end
    abspath(path)
end
