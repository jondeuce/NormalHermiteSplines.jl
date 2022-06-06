using Test
using NormalHermiteSplines
using NormalHermiteSplines: ElasticCholesky, ElasticNormalSpline

using Aqua
using DoubleFloats
using LinearAlgebra
using Random
using StaticArrays
using UnPack

@testset "NormalHermiteSplines.jl" begin
    include("1D.jl")
    include("2D.jl")
    include("3D.jl")
    include("elastic.jl")
end

@testset "Aqua tests" begin
    Aqua.test_all(NormalHermiteSplines)
end
