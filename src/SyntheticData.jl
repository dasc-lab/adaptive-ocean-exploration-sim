module SyntheticData

using AbstractGPs, KernelFunctions, Random, StaticArrays, LinearAlgebra

# Generate spatial Matern12 synthetic data
# TODO: Currently defaults to ZeroMean Mean Function, overload to include other mean functions
function matern12_spatial(dim, xstart, xstop, ystart, ystop, σ_sq, λₓ)
    # Create the spatial kernel
    spatial_kernel = σ_sq * Matern12Kernel() ∘ ScaleTransform(λₓ);
    
    # Define the mean function
    mean_function = AbstractGPs.ZeroMean();
    
    # Create the Guassian process with the spatial kernel
    gp = AbstractGPs.GP(mean_function, spatial_kernel);

    # Define the spatial grid
    x = range(xstart, stop=xstop, length=dim);
    y = range(ystart, stop=ystop, length=dim);
    spatial_points = vec([SVector(xi, yi) for xi in x, yi in y])

    # Generate synthetic data by sampling from the GP
    y = rand(gp(spatial_points));

    # # If you want reproducible data
    # rng = MersenneTwiser(123);
    # y = rand(rng, gp(spatial_points));

    # Reshape the data into an array
    data_2d = reshape(y, dim, dim);
    
    return data_2d
end

# Generate spatiotemporal Matern12 synthetic data
# TODO: Currently defaults to ZeroMean Mean Function, overload to include other mean functions
function matern12_spatiotemporal(sdim, tdim, xstart, xstop, ystart, ystop, tstart, tstop, σ_sq, λₓ, λₜ)
    # Create the kernels
    spatial_kernel = Matern12Kernel() ∘ ScaleTransform(λₓ);
    temporal_kernel = Matern12Kernel() ∘ ScaleTransform(λₜ);
    
    # Create spatiotemporal kernel
    spatiotemporal_kernel = σ_sq * spatial_kernel * temporal_kernel;

    # Define the mean function
    mean_function = AbstractGPs.ZeroMean();
    
    # Create the Guassian process with the spatial kernel
    gp = AbstractGPs.GP(mean_function, spatiotemporal_kernel);

    # Define the spatial grid
    x = range(xstart, stop=xstop, length=sdim);
    y = range(ystart, stop=ystop, length=sdim);
    spatial_points = vec([SVector(xi, yi) for xi in x, yi in y]);

    # Define temporal points
    temporal_points = range(tstart, stop=tstop, length=tdim);

    # Create a grid of spatiotemporal points
    spatiotemporal_points = vec([SVector(spatial[1], spatial[2], t) for spatial in spatial_points, t in temporal_points]);

    # Generate synthetic data by sampling from the GP
    y = rand(gp(spatiotemporal_points));

    # # If you want reproducible data
    # rng = MersenneTwister(123);
    # y = rand(rng, gp(spatiotemporal_points))

    # Reshape the data into a 3D array (spatial_x, spatial_y, temporal)
    data_3d = reshape(y, sdim, sdim, tdim);
    
    return data_3d
end

end # module SyntheticData