abstract type ReproducingKernel end
abstract type ReproducingKernel_0 <: ReproducingKernel end
abstract type ReproducingKernel_1 <: ReproducingKernel_0 end
abstract type ReproducingKernel_2 <: ReproducingKernel_1 end

@doc raw"
`struct RK_H0{T} <: ReproducingKernel_0`

Defines a type of reproducing kernel of Bessel Potential space ``H^{n/2 + 1/2}_ε (R^n)`` ('Basic Matérn kernel'):
```math
V(\eta , \xi, \varepsilon) = \exp (-\varepsilon |\xi - \eta|) \, .
```
# Fields
- `ε::T`: 'scaling parameter' from the Bessel Potential space definition,
           it may be omitted in the struct constructor otherwise it must be greater than zero
"
struct RK_H0{T} <: ReproducingKernel_0
    ε::T
    RK_H0() = new{Float64}(0.0)
    function RK_H0(ε)
        @assert ε > 0
        return new{typeof(float(ε))}(float(ε))
    end
    function RK_H0{T}(ε) where {T}
        @assert ε > 0
        return new{T}(T(ε))
    end
end
Base.eltype(::RK_H0{T}) where {T} = T

@doc raw"
`struct RK_H1{T} <: ReproducingKernel_1`

Defines a type of reproducing kernel of Bessel Potential space ``H^{n/2 + 3/2}_ε (R^n)`` ('Linear Matérn kernel'):
```math
V(\eta , \xi, \varepsilon) = \exp (-\varepsilon |\xi - \eta|)
             (1 + \varepsilon |\xi  - \eta|) \, .
```
# Fields
- `ε::T`: 'scaling parameter' from the Bessel Potential space definition,
           it may be omitted in the struct constructor otherwise it must be greater than zero
"
struct RK_H1{T} <: ReproducingKernel_1
    ε::T
    RK_H1() = new{Float64}(0.0)
    function RK_H1(ε)
        @assert ε > 0
        return new{typeof(float(ε))}(float(ε))
    end
    function RK_H1{T}(ε) where {T}
        @assert ε > 0
        return new{T}(T(ε))
    end
end
Base.eltype(::RK_H1{T}) where {T} = T

@doc raw"
`struct RK_H2{T} <: ReproducingKernel_2`

Defines a type of reproducing kernel of Bessel Potential space ``H^{n/2 + 5/2}_ε (R^n)`` ('Quadratic Matérn kernel'):
```math
V(\eta , \xi, \varepsilon) = \exp (-\varepsilon |\xi - \eta|)
             (3 + 3\varepsilon |\xi  - \eta| + \varepsilon ^2 |\xi - \eta| ^2 ) \, .
```
# Fields
- `ε::T`: 'scaling parameter' from the Bessel Potential space definition,
           it may be omitted in the struct constructor otherwise it must be greater than zero
"
struct RK_H2{T} <: ReproducingKernel_2
    ε::T
    RK_H2() = new{Float64}(0.0)
    function RK_H2(ε)
        @assert ε > 0
        return new{typeof(float(ε))}(float(ε))
    end
    function RK_H2{T}(ε) where {T}
        @assert ε > 0
        return new{T}(T(ε))
    end
end
Base.eltype(::RK_H2{T}) where {T} = T

@inline _norm(x::SVector) = norm(x)
@inline _norm(x::SVector{1}) = abs(x[1])

@inline @fastmath function _rk(kernel::RK_H2, η::SVector, ξ::SVector)
    x = kernel.ε * _norm(η - ξ)
    return (3 + x * (3 + x)) * exp(-x)
end

@inline @fastmath function _rk(kernel::RK_H1, η::SVector, ξ::SVector)
    x = kernel.ε * _norm(η - ξ)
    return (1 + x) * exp(-x)
end

@inline @fastmath function _rk(kernel::RK_H0, η::SVector, ξ::SVector)
    x = kernel.ε * _norm(η - ξ)
    return exp(-x)
end

@inline @fastmath function _∂rk_∂e(kernel::RK_H2, η::SVector, ξ::SVector, e::SVector)
    t = η - ξ
    x = kernel.ε * _norm(t)
    return kernel.ε^2 * exp(-x) * (1 + x) * (t ⋅ e)
end

@inline @fastmath function _∂rk_∂e(kernel::RK_H1, η::SVector, ξ::SVector, e::SVector)
    t = η - ξ
    x = kernel.ε * _norm(t)
    return kernel.ε^2 * exp(-x) * (t ⋅ e)
end

@inline @fastmath function _∂rk_∂η(kernel::RK_H2, η::SVector, ξ::SVector)
    t = η - ξ
    x = kernel.ε * _norm(t)
    return -kernel.ε^2 * exp(-x) * (1 + x) * t
end

@inline @fastmath function _∂rk_∂η(kernel::RK_H1, η::SVector, ξ::SVector)
    t = η - ξ
    x = kernel.ε * _norm(t)
    return -kernel.ε^2 * exp(-x) * t
end

@inline @fastmath function _∂rk_∂η(kernel::RK_H0, η::SVector, ξ::SVector)
    # Note: Derivative of spline built with reproducing kernel RK_H0 does not exist at the spline nodes
    t    = η - ξ
    tnrm = _norm(t)
    x    = kernel.ε * tnrm
    ∇    = kernel.ε * exp(-x)
    t    = ifelse(x > eps(typeof(x)), t, zeros(t))
    ∇    *= ifelse(x > eps(typeof(x)), inv(tnrm), one(tnrm))
    ∇    *= t
    return ∇
end

@inline @fastmath function _∂²rk_∂²e(kernel::RK_H2, η::SVector{n}, ξ::SVector{n}, êη::SVector{n}, êξ::SVector{n}) where {n}
    ε     = kernel.ε
    ε²    = ε * ε
    t     = η - ξ
    tnrm  = _norm(t)
    x     = ε * tnrm
    ε²e⁻ˣ = ε² * exp(-x)
    ∇²    = ((1 + x) * ε²e⁻ˣ) * (êξ ⋅ êη)
    ∇²    -= (ε² * ε²e⁻ˣ) * (êξ ⋅ t) * (t ⋅ êη)
    return ∇²
end

@inline @fastmath function _∂²rk_∂²e(kernel::RK_H1, η::SVector{n}, ξ::SVector{n}, êη::SVector{n}, êξ::SVector{n}) where {n}
    # Note: Second derivative of spline built with reproducing kernel RK_H1 does not exist at the spline nodes
    ε     = kernel.ε
    ε²    = ε * ε
    t     = η - ξ
    tnrm  = _norm(t)
    x     = ε * tnrm
    ε²e⁻ˣ = ε² * exp(-x)
    ∇²    = ε²e⁻ˣ * (êξ ⋅ êη)
    ∇²    -= ifelse(x > eps(typeof(x)), (ε * ε²e⁻ˣ / tnrm) * (êξ ⋅ t) * (t ⋅ êη), zero(∇²))
    return ∇²
end

@inline @fastmath function _∂²rk_∂η∂ξ(kernel::RK_H2, η::SVector{n}, ξ::SVector{n}) where {n}
    ε     = kernel.ε
    ε²    = ε * ε
    t     = η - ξ
    tnrm  = _norm(t)
    x     = ε * tnrm
    ε²e⁻ˣ = ε² * exp(-x)
    S     = SMatrix{n, n, typeof(x)}
    ∇²    = S(((1 + x) * ε²e⁻ˣ) * LinearAlgebra.I)
    ∇²    -= ((ε² * ε²e⁻ˣ) * t) * t'
    return ∇²
end

@inline @fastmath function _∂²rk_∂η∂ξ(kernel::RK_H1, η::SVector{n}, ξ::SVector{n}) where {n}
    # Note: Second derivative of spline built with reproducing kernel RK_H1 does not exist at the spline nodes
    ε     = kernel.ε
    ε²    = ε * ε
    t     = η - ξ
    tnrm  = _norm(t)
    x     = ε * tnrm
    ε²e⁻ˣ = ε² * exp(-x)
    S     = SMatrix{n, n, typeof(x)}
    ∇²    = S(ε²e⁻ˣ * LinearAlgebra.I)
    ∇²    -= ifelse(x > eps(typeof(x)), ((ε * ε²e⁻ˣ / tnrm) * t) * t', zeros(S))
    return ∇²
end
