using Test

using Simulator.Utilities

@testset "pos2coord" begin
    @test Utilities.pos2coord(3,5) == 8
    @test Utilities.pos2coord(3.5,4.5) == 8
    @test Utilities.pos2coord(3.5,4.5) == 8.0
end