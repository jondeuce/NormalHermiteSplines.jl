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
Base.eltype(::Type{RK_H0{T}}) where {T} = T

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
Base.eltype(::Type{RK_H1{T}}) where {T} = T

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
Base.eltype(::Type{RK_H2{T}}) where {T} = T

@inline _norm(x::SVector) = norm(x)
@inline _norm(x::SVector{1}) = abs(x[1])

@inline @fastmath @muladd function _rk(kernel::RK_H2, η::SVector, ξ::SVector)
    x = kernel.ε * _norm(η - ξ)
    return (3 + x * (3 + x)) * exp(-x)
end

@inline @fastmath @muladd function _rk(kernel::RK_H1, η::SVector, ξ::SVector)
    x = kernel.ε * _norm(η - ξ)
    return (1 + x) * exp(-x)
end

@inline @fastmath @muladd function _rk(kernel::RK_H0, η::SVector, ξ::SVector)
    x = kernel.ε * _norm(η - ξ)
    return exp(-x)
end

@inline @fastmath @muladd function _∂rk_∂ηⁱ_ûᵢ(kernel::RK_H2, η::SVector, ξ::SVector, û::SVector)
    t = η - ξ
    x = kernel.ε * _norm(t)
    return -kernel.ε^2 * exp(-x) * (1 + x) * (t ⋅ û)
end

@inline @fastmath @muladd function _∂rk_∂ηⁱ_ûᵢ(kernel::RK_H1, η::SVector, ξ::SVector, û::SVector)
    t = η - ξ
    x = kernel.ε * _norm(t)
    return -kernel.ε^2 * exp(-x) * (t ⋅ û)
end

@inline @fastmath @muladd function _∂rk_∂ηⁱ_ûᵢ(kernel::RK_H0, η::SVector, ξ::SVector, û::SVector)
    t     = η - ξ
    tnrm  = _norm(t)
    itnrm = ifelse(tnrm > eps(typeof(tnrm)), inv(tnrm), zero(tnrm))
    x     = kernel.ε * tnrm
    return -kernel.ε * exp(-x) * itnrm * (t ⋅ û)
end

@inline @fastmath @muladd function _∂rk_∂η(kernel::RK_H2, η::SVector, ξ::SVector)
    t = η - ξ
    x = kernel.ε * _norm(t)
    return -kernel.ε^2 * exp(-x) * (1 + x) * t
end

@inline @fastmath @muladd function _∂rk_∂η(kernel::RK_H1, η::SVector, ξ::SVector)
    t = η - ξ
    x = kernel.ε * _norm(t)
    return -kernel.ε^2 * exp(-x) * t
end

@inline @fastmath @muladd function _∂rk_∂η(kernel::RK_H0, η::SVector, ξ::SVector)
    # Note: Derivative of spline built with reproducing kernel RK_H0 is discontinuous at the spline nodes, i.e. when η = ξ.
    t     = η - ξ
    tnrm  = _norm(t)
    itnrm = ifelse(tnrm > eps(typeof(tnrm)), inv(tnrm), zero(tnrm))
    x     = kernel.ε * tnrm
    ∇     = (-kernel.ε * exp(-x) * itnrm) * t
    return ∇
end

@inline @fastmath @muladd function _rk_with_∂rk_∂η(kernel::RK_H2, η::SVector, ξ::SVector)
    t   = η - ξ
    x   = kernel.ε * _norm(t)
    e⁻ˣ = exp(-x)
    y   = (3 + x * (3 + x)) * e⁻ˣ
    ∇   = -kernel.ε^2 * e⁻ˣ * (1 + x) * t
    return y, ∇
end

@inline @fastmath @muladd function _rk_with_∂rk_∂η(kernel::RK_H1, η::SVector, ξ::SVector)
    t   = η - ξ
    x   = kernel.ε * _norm(t)
    e⁻ˣ = exp(-x)
    y   = (1 + x) * e⁻ˣ
    ∇   = -kernel.ε^2 * e⁻ˣ * t
    return y, ∇
end

@inline @fastmath @muladd function _rk_with_∂rk_∂η(kernel::RK_H0, η::SVector, ξ::SVector)
    # Note: Derivative of spline built with reproducing kernel RK_H0 is discontinuous at the spline nodes, i.e. when η = ξ.
    t     = η - ξ
    tnrm  = _norm(t)
    itnrm = ifelse(tnrm > eps(typeof(tnrm)), inv(tnrm), zero(tnrm))
    x     = kernel.ε * tnrm
    y     = exp(-x)
    ∇     = (-kernel.ε * y * itnrm) * t
    return y, ∇
end

@inline @fastmath @muladd function _∂²rk_∂ηⁱ∂ξʲ_ûᵢ_v̂ⱼ(kernel::RK_H2, η::SVector{n}, ξ::SVector{n}, û::SVector{n}, v̂::SVector{n}) where {n}
    ε     = kernel.ε
    ε²    = ε * ε
    t     = η - ξ
    tnrm  = _norm(t)
    x     = ε * tnrm
    ε²e⁻ˣ = ε² * exp(-x)
    return ((1 + x) * ε²e⁻ˣ) * (û ⋅ v̂) - (ε² * ε²e⁻ˣ) * (û ⋅ t) * (t ⋅ v̂)
end

@inline @fastmath @muladd function _∂²rk_∂ηⁱ∂ξʲ_ûᵢ_v̂ⱼ(kernel::RK_H1, η::SVector{n}, ξ::SVector{n}, û::SVector{n}, v̂::SVector{n}) where {n}
    # Note: Second derivative of spline built with reproducing kernel RK_H1 is discontinuous at the spline nodes, i.e. when η = ξ.
    ε     = kernel.ε
    ε²    = ε * ε
    t     = η - ξ
    tnrm  = _norm(t)
    itnrm = ifelse(tnrm > eps(one(tnrm)), inv(tnrm), zero(tnrm))
    x     = ε * tnrm
    ε²e⁻ˣ = ε² * exp(-x)
    return ε²e⁻ˣ * (û ⋅ v̂) - (ε * ε²e⁻ˣ * itnrm) * (û ⋅ t) * (t ⋅ v̂)
end

@inline @fastmath @muladd function _∂²rk_∂ηⁱ∂ξ_ûᵢ(kernel::RK_H2, η::SVector{n}, ξ::SVector{n}, û::SVector{n}) where {n}
    ε     = kernel.ε
    ε²    = ε * ε
    t     = η - ξ
    tnrm  = _norm(t)
    x     = ε * tnrm
    ε²e⁻ˣ = ε² * exp(-x)
    return ((1 + x) * ε²e⁻ˣ) * û - ((ε² * ε²e⁻ˣ) * (û ⋅ t)) * t
end

@inline @fastmath @muladd function _∂²rk_∂ηⁱ∂ξ_ûᵢ(kernel::RK_H1, η::SVector{n}, ξ::SVector{n}, û::SVector{n}) where {n}
    # Note: Second derivative of spline built with reproducing kernel RK_H1 is discontinuous at the spline nodes, i.e. when η = ξ.
    ε     = kernel.ε
    ε²    = ε * ε
    t     = η - ξ
    tnrm  = _norm(t)
    itnrm = ifelse(tnrm > eps(one(tnrm)), inv(tnrm), zero(tnrm))
    x     = ε * tnrm
    ε²e⁻ˣ = ε² * exp(-x)
    return ε²e⁻ˣ * û - ((ε * ε²e⁻ˣ * itnrm) * (û ⋅ t)) * t
end

@inline @fastmath @muladd function _∂rk_∂ηⁱ_ûᵢ_with_∂²rk_∂ηⁱ∂ξ_ûᵢ(kernel::RK_H2, η::SVector{n}, ξ::SVector{n}, û::SVector{n}) where {n}
    ε     = kernel.ε
    ε²    = ε * ε
    t     = η - ξ
    tnrm  = _norm(t)
    x     = ε * tnrm
    ε²e⁻ˣ = ε² * exp(-x)
    tmp1  = ε²e⁻ˣ * û
    tmp2  = (1 + x) * tmp1
    ∇ᵀû  = -(tmp2 ⋅ t)
    ∇²û  = tmp2 - ε² * (tmp1 ⋅ t) * t
    return ∇ᵀû, ∇²û
end

@inline @fastmath @muladd function _∂rk_∂ηⁱ_ûᵢ_with_∂²rk_∂ηⁱ∂ξ_ûᵢ(kernel::RK_H1, η::SVector{n}, ξ::SVector{n}, û::SVector{n}) where {n}
    # Note: Second derivative of spline built with reproducing kernel RK_H1 is discontinuous at the spline nodes, i.e. when η = ξ.
    ε     = kernel.ε
    ε²    = ε * ε
    t     = η - ξ
    tnrm  = _norm(t)
    itnrm = ifelse(tnrm > eps(one(tnrm)), inv(tnrm), zero(tnrm))
    x     = ε * tnrm
    ε²e⁻ˣ = ε² * exp(-x)
    tmp1  = ε²e⁻ˣ * û
    tmp2  = (tmp1 ⋅ t)
    ∇ᵀû  = -tmp2
    ∇²û  = tmp1 - (ε * itnrm * tmp2) * t
    return ∇ᵀû, ∇²û
end

@inline @fastmath @muladd function _∂²rk_∂η∂ξ(kernel::RK_H2, η::SVector{n}, ξ::SVector{n}) where {n}
    ε     = kernel.ε
    ε²    = ε * ε
    t     = η - ξ
    tnrm  = _norm(t)
    x     = ε * tnrm
    ε²e⁻ˣ = ε² * exp(-x)
    ∇²    = SMatrix{n, n, typeof(x)}(((1 + x) * ε²e⁻ˣ) * LinearAlgebra.I)
    ∇²    -= ((ε² * ε²e⁻ˣ) * t) * t'
    return ∇²
end

@inline @fastmath @muladd function _∂²rk_∂η∂ξ(kernel::RK_H1, η::SVector{n}, ξ::SVector{n}) where {n}
    # Note: Second derivative of spline built with reproducing kernel RK_H1 is discontinuous at the spline nodes, i.e. when η = ξ.
    ε     = kernel.ε
    ε²    = ε * ε
    t     = η - ξ
    tnrm  = _norm(t)
    itnrm = ifelse(tnrm > eps(one(tnrm)), inv(tnrm), zero(tnrm))
    x     = ε * tnrm
    ε²e⁻ˣ = ε² * exp(-x)
    ∇²    = SMatrix{n, n, typeof(x)}(ε²e⁻ˣ * LinearAlgebra.I)
    ∇²    -= ((ε * ε²e⁻ˣ * itnrm) * t) * t'
    return ∇²
end
