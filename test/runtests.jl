using Test
using NormalHermiteSplines
using NormalHermiteSplines: ElasticCholesky, ElasticNormalSpline

using DoubleFloats
using LinearAlgebra
using Random
using StaticArrays

@testset "NormalHermiteSplines.jl" begin
    include("1D.jl")
    include("2D.jl")
    include("3D.jl")
    include("elastic.jl")
end
