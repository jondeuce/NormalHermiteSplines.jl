@doc raw"
`struct RK_H0{T <: AbstractFloat} <: ReproducingKernel_0`

Defines a type of reproducing kernel of Bessel Potential space ``H^{n/2 + 1/2}_ε (R^n)`` ('Basic Matérn kernel'):
```math
V(\eta , \xi, \varepsilon) = \exp (-\varepsilon |\xi - \eta|) \, .
```
# Fields
- `ε::T`: 'scaling parameter' from the Bessel Potential space definition,
           it may be omitted in the struct constructor otherwise it must be greater than zero
"
struct RK_H0{T <: AbstractFloat} <: ReproducingKernel_0
     ε::T
     RK_H0() = new{Float64}(0.0)
     function RK_H0(ε::T) where T <: AbstractFloat
        if ε <= 0
          throw(DomainError(ε, "Parameter ε must be a positive number."))
        end
        new{T}(ε)
     end
     function RK_H0(ε::Integer)
        if ε <= 0
           throw(DomainError(ε, "Parameter ε must be a positive number."))
        end
        new{Float64}(convert(Float64, ε))
     end
end

@doc raw"
`struct RK_H1{T <: AbstractFloat} <: ReproducingKernel_1`

Defines a type of reproducing kernel of Bessel Potential space ``H^{n/2 + 3/2}_ε (R^n)`` ('Linear Matérn kernel'):
```math
V(\eta , \xi, \varepsilon) = \exp (-\varepsilon |\xi - \eta|)
             (1 + \varepsilon |\xi  - \eta|) \, .
```
# Fields
- `ε::T`: 'scaling parameter' from the Bessel Potential space definition,
           it may be omitted in the struct constructor otherwise it must be greater than zero
"
struct RK_H1{T <: AbstractFloat} <: ReproducingKernel_1
     ε::T
     RK_H1() = new{Float64}(0.0)
     function RK_H1(ε::T) where T <: AbstractFloat
        if ε <= 0
           throw(DomainError(ε, "Parameter ε must be a positive number."))
        end
        new{T}(ε)
     end
     function RK_H1(ε::Integer)
        if ε <= 0
           throw(DomainError(ε, "Parameter ε must be a positive number."))
        end
        new{Float64}(convert(Float64, ε))
     end
end

@doc raw"
`struct RK_H2{T <: AbstractFloat} <: ReproducingKernel_2`

Defines a type of reproducing kernel of Bessel Potential space ``H^{n/2 + 5/2}_ε (R^n)`` ('Quadratic Matérn kernel'):
```math
V(\eta , \xi, \varepsilon) = \exp (-\varepsilon |\xi - \eta|)
             (3 + 3\varepsilon |\xi  - \eta| + \varepsilon ^2 |\xi - \eta| ^2 ) \, .
```
# Fields
- `ε::T`: 'scaling parameter' from the Bessel Potential space definition,
           it may be omitted in the struct constructor otherwise it must be greater than zero
"
struct RK_H2{T <: AbstractFloat} <: ReproducingKernel_2
     ε::T
     RK_H2() = new{Float64}(0.0)
     function RK_H2(ε::T) where T <: AbstractFloat
        if ε <= 0
           throw(DomainError(ε, "Parameter ε must be a positive number."))
        end
        new{T}(ε)
     end
     function RK_H2(ε::Integer)
        if ε <= 0
           throw(DomainError(ε, "Parameter ε must be a positive number."))
        end
        new{Float64}(convert(Float64, ε))
     end
end

@inline function _rk(kernel::RK_H2, η::AbstractVector, ξ::AbstractVector)
   x = kernel.ε * norm(ξ .- η)
   return (3 + x * (3 + x)) * exp(-x)
end

@inline function _rk(kernel::RK_H1, η::AbstractVector, ξ::AbstractVector)
   x = kernel.ε * norm(ξ .- η)
   return (1 + x) * exp(-x)
end

@inline function _rk(kernel::RK_H0, η::AbstractVector, ξ::AbstractVector)
   x = kernel.ε * norm(ξ .- η)
   return exp(-x)
end

@inline function _∂rk_∂e(kernel::RK_H2, η::AbstractVector, ξ::AbstractVector, e::AbstractVector)
   t = η .- ξ
   x = kernel.ε * norm(t)
   s = sum(t .* e)
   return kernel.ε^2 * exp(-x) * (1 + x) * s
end

@inline function _∂rk_∂e(kernel::RK_H1, η::AbstractVector, ξ::AbstractVector, e::AbstractVector)
   t = η .- ξ
   x = kernel.ε * norm(t)
   s = sum(t .* e)
   return kernel.ε^2 * exp(-x) * s
end

@inline function _∂rk_∂η_k(kernel::RK_H2, η::AbstractVector, ξ::AbstractVector, k::Int)
   x = kernel.ε * norm(η .- ξ)
   return kernel.ε^2 * exp(-x) * (1 + x) * (ξ[k] - η[k])
end

@inline function _∂rk_∂η_k(kernel::RK_H1, η::AbstractVector, ξ::AbstractVector, k::Int)
   x = kernel.ε * norm(η .- ξ)
   return kernel.ε^2 * exp(-x) * (ξ[k] - η[k])
end

@inline function _∂rk_∂η_k(kernel::RK_H0, η::AbstractVector, ξ::AbstractVector, k::Int)
   # Note: Derivative of spline built with reproducing kernel RK_H0 does not exist at the spline nodes
   normt = norm(η - ξ)
   x = kernel.ε * normt
   return normt < sqrt(eps(typeof(x))) ?
      kernel.ε * exp(-x) * sign(ξ[k] - η[k]) :
      kernel.ε * exp(-x) * (ξ[k] - η[k]) / normt
end

@inline function _∂²rk_∂η_r_∂ξ_k(kernel::RK_H2, η::AbstractVector, ξ::AbstractVector, r::Int, k::Int)
   x = kernel.ε * norm(η .- ξ)
   if r == k
      if x > 0
         kernel.ε^2 * exp(-x) * (1 + x - (kernel.ε * (ξ[r] - η[r]))^2)
      else
         kernel.ε^2
      end
   else
      if x > 0
         -kernel.ε^4 * exp(-x) * (ξ[r] - η[r]) * (ξ[k] - η[k])
      else
         zero(x)
      end
   end
end

@inline function _∂²rk_∂η_r_∂ξ_k(kernel::RK_H1, η::AbstractVector, ξ::AbstractVector, r::Int, k::Int)
   t = norm(η .- ξ)
   x = kernel.ε * t
   if r == k
      if t > 0
         kernel.ε^2 * exp(-x) * (1 - kernel.ε * (ξ[r] - η[r])^2 / t)
      else
         kernel.ε^2
      end
   else
      if t > 0
         -kernel.ε^3 * exp(-x) * (ξ[r] - η[r]) * (ξ[k] - η[k]) / t
      else
         zero(x)
      end
   end
end
