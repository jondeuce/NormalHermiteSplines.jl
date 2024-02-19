using Test
using NormalHermiteSplines
using NormalHermiteSplines:
    AbstractNormalSpline, ReproducingKernel_0, ReproducingKernel_1, ReproducingKernel_2,
    _evaluate, _evaluate_gradient, _evaluate_with_gradient,
    _rk, _∂rk_∂η, _∂rk_∂ηⁱ_ûᵢ, _∂²rk_∂η∂ξ, _∂²rk_∂ηⁱ∂ξ_ûᵢ, _∂²rk_∂ηⁱ∂ξʲ_ûᵢ_v̂ⱼ

using Aqua
using DoubleFloats
using ForwardDiff
using LinearAlgebra
using Random
using StaticArrays

@testset "NormalHermiteSplines.jl" begin
    include("1D.jl")
    include("2D.jl")
    include("3D.jl")
    include("elastic.jl")
    include("derivatives.jl")
end

@testset "Aqua tests" begin
    Aqua.test_all(NormalHermiteSplines)
end
