using Plots

include("src/jordan_lake_domain.jl")

plot_object = plot(JordanLakeDomain.polygon, aspect_ratio=:equal)
plot_object = xlims!(plot_object, -1, 2.5)
for p in JordanLakeDomain.𝐬
    plot_object = scatter!(plot_object, [p[1]], [p[2]],
        color=:green, ms=2, legend=false)
end
plot_object = plot!(plot_object, size=(375, 600),
    xlabel="x [km]", ylabel="y [km]")
display(plot_object)
