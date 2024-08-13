using Plots
plotlyjs()

include("src/jordan_lake_domain.jl")

plot_object = plot(JordanLakeDomain.polygon, aspect_ratio=:equal)
plot_object = xlims!(plot_object, -1, 1)
location_index = 1
for p in JordanLakeDomain.𝐬
    plot_object = scatter!(plot_object, [p[1]], [p[2]],
        color=:green, ms=2, legend=false, hover="$location_index")
    location_index += 1
end
plot_object = plot!(plot_object, size=(312.5, 500),
    xlabel="x [km]", ylabel="y [km]")
display(plot_object)
