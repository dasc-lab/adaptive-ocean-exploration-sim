using Test

using Simulator.Utilities

@testset "pos2coord" begin
    @test Utilities.pos2coord(3,5) == 8
    @test Utilities.pos2coord(3.5,4.5) == 8
    @test Utilities.pos2coord(3.5,4.5) == 8.0
end

@testset "gps2coord" begin
    @test Utilities.gps2coord([22.22, 22.22], [22.23, 22.22]) ≈ [0, 1.100] atol=0.01
    @test Utilities.gps2coord([40.7128, -74.0060], [34.0522, -118.2437]) ≈ [-3690, 240] atol=10
end