using Test

using Simulator.Utilities

@testset "pos2coord" begin
    @test Utilities.pos2coord(3,5) == 8
    @test Utilities.pos2coord(3.5,4.5) == 8
    @test Utilities.pos2coord(3.5,4.5) == 8.0
end

@testset "gps2coord" begin
    @test Utilities.gps2coord([22.22, 22.22], [22.23, 22.22]) ≈ [0, 1.100] atol=0.01
    @test Utilities.gps2coord([35.7708098, -79.0247384], [35.7459328, -79.0294959]) ≈ [-0.5, -2.8] atol=0.1
end