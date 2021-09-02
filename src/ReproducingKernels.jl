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
end
function RK_H0(ε)
    ε <= 0 && throw(DomainError(ε, "Parameter ε must be a positive number."))
    ε = float(ε)
    RK_H0{typeof(ε)}(ε)
end
RK_H0() = RK_H0{Float64}(0.0)
RK_H0{T}() where {T} = RK_H0{T}(zero(T))

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
end
function RK_H1(ε)
    ε <= 0 && throw(DomainError(ε, "Parameter ε must be a positive number."))
    ε = float(ε)
    RK_H1{typeof(ε)}(ε)
end
RK_H1() = RK_H1{Float64}(0.0)
RK_H1{T}() where {T} = RK_H1{T}(zero(T))

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
end
function RK_H2(ε)
    ε <= 0 && throw(DomainError(ε, "Parameter ε must be a positive number."))
    ε = float(ε)
    RK_H2{typeof(ε)}(ε)
end
RK_H2() = RK_H2{Float64}(0.0)
RK_H2{T}() where {T} = RK_H2{T}(zero(T))

@inline function _rk(kernel::RK_H2, η::SVector, ξ::SVector)
    x = kernel.ε * norm(η - ξ)
    return (3 + x * (3 + x)) * exp(-x)
end

@inline function _rk(kernel::RK_H1, η::SVector, ξ::SVector)
    x = kernel.ε * norm(η - ξ)
    return (1 + x) * exp(-x)
end

@inline function _rk(kernel::RK_H0, η::SVector, ξ::SVector)
    x = kernel.ε * norm(η - ξ)
    return exp(-x)
end

@inline function _∂rk_∂e(kernel::RK_H2, η::SVector, ξ::SVector, e::SVector)
    t = η - ξ
    x = kernel.ε * norm(t)
    return kernel.ε^2 * exp(-x) * (1 + x) * (t ⋅ e)
end

@inline function _∂rk_∂e(kernel::RK_H1, η::SVector, ξ::SVector, e::SVector)
    t = η - ξ
    x = kernel.ε * norm(t)
    return kernel.ε^2 * exp(-x) * (t ⋅ e)
end

@inline function _∂rk_∂η_k(kernel::RK_H2, η::SVector, ξ::SVector, ::Val{k}) where {k}
    x = kernel.ε * norm(η - ξ)
    return kernel.ε^2 * exp(-x) * (1 + x) * (ξ[k] - η[k])
end

@inline function _∂rk_∂η_k(kernel::RK_H1, η::SVector, ξ::SVector, ::Val{k}) where {k}
    x = kernel.ε * norm(η - ξ)
    return kernel.ε^2 * exp(-x) * (ξ[k] - η[k])
end

@inline function _∂rk_∂η_k(kernel::RK_H0, η::SVector, ξ::SVector, ::Val{k}) where {k}
    # Note: Derivative of spline built with reproducing kernel RK_H0 does not exist at the spline nodes
    normt = norm(η - ξ)
    x = kernel.ε * normt
    ∇ = kernel.ε * exp(-x)
    return normt < sqrt(eps(typeof(x))) ? ∇ * sign(ξ[k] - η[k]) : ∇ * (ξ[k] - η[k]) / normt
end

@generated function _∂rk_∂η(kernel::RK, η::SVector{n}, ξ::SVector{n}) where {n, RK <: ReproducingKernel_0}
    vals = [:(_∂rk_∂η_k(kernel, η, ξ, Val($k))) for k in 1:n]
    :(Base.@_inline_meta; SVector{$n}(tuple($(vals...))))
end

@inline function _∂²rk_∂η_r_∂ξ_k(kernel::RK_H2, η::SVector, ξ::SVector, ::Val{r}, ::Val{k}) where {r,k}
    ε  = kernel.ε
    ε² = ε * ε
    x  = ε * norm(η - ξ)
    if r == k
        ε² * ifelse(x <= 0, one(x), exp(-x) * (1 + x - (ε * (ξ[r] - η[r]))^2))
    else
        ifelse(x <= 0, zero(x), -ε² * ε² * exp(-x) * (ξ[r] - η[r]) * (ξ[k] - η[k]))
    end
end

@inline function _∂²rk_∂η_r_∂ξ_k(kernel::RK_H1, η::SVector, ξ::SVector, ::Val{r}, ::Val{k}) where {r,k}
    ε  = kernel.ε
    ε² = ε * ε
    t  = norm(η - ξ)
    x  = ε * t
    if r == k
        ifelse(t <= 0, ε², ε² * exp(-x) * (1 - ε * (ξ[r] - η[r])^2 / t))
    else
        ifelse(t <= 0, zero(x), -ε * ε² * exp(-x) * (ξ[r] - η[r]) * (ξ[k] - η[k]) / t)
    end
end

@generated function _∂²rk_∂η∂ξ(kernel::RK, η::SVector{n}, ξ::SVector{n}) where {n, RK <: ReproducingKernel_1}
    vals = [:(_∂²rk_∂η_r_∂ξ_k(kernel, η, ξ, Val($r), Val($k))) for r in 1:n, k in 1:n]
    :(Base.@_inline_meta; SMatrix{$n,$n}(tuple($(vals...))))
end
