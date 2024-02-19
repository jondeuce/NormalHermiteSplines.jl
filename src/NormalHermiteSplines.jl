module NormalHermiteSplines

export prepare, construct, interpolate
export evaluate, evaluate!, evaluate_one, evaluate_gradient, evaluate_derivative
export NormalSpline, ElasticCholesky, ElasticNormalSpline, RK_H0, RK_H1, RK_H2
export get_epsilon, estimate_epsilon, get_cond, estimate_cond, estimate_accuracy

using LinearAlgebra: LinearAlgebra, Cholesky, Factorization, Hermitian, UpperTriangular, cholesky, cholesky!, ldiv!, norm, ⋅
using MuladdMacro: @muladd
using StaticArrays: StaticArrays, SMatrix, SVector
using UnsafeArrays: UnsafeArrays, uview

const AbstractArrOfSVecs{n, T, D} = AbstractArray{S, D} where {S <: SVector{n, T}}
const AbstractVecOfSVecs{n, T} = AbstractArrOfSVecs{n, T, 1}
const VecOfSVecs{n, T} = Vector{SVector{n, T}}

@inline svectors(x::AbstractMatrix{T}) where {T} = reinterpret(reshape, SVector{size(x, 1), T}, x)
@inline svectors(x::AbstractVector{T}) where {T} = reinterpret(SVector{1, T}, x)

include("ReproducingKernels.jl")
include("GramMatrix.jl")
include("Splines.jl")
include("Utils.jl")
include("Interpolate.jl")

####
#### Public API
####

#### ReproducingKernel_0

"""
`prepare(nodes::AbstractMatrix{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Prepare the spline by constructing and factoring a Gram matrix of the interpolation problem.
Initialize the `NormalSpline` object.

# Arguments

  - `nodes`: The function value nodes.
    This should be an `n×n_1` matrix, where `n` is dimension of the sampled space and
    `n₁` is the number of function value nodes. It means that each column in the matrix defines one node.
  - `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
    It must be a struct object of the following type:
    `RK_H0` if the spline is constructing as a continuous function,
    `RK_H1` if the spline is constructing as a differentiable function,
    `RK_H2` if the spline is constructing as a twice differentiable function.

# Returns

  - The partly initialized `NormalSpline` object that must be passed to `construct` function in order to complete the spline initialization.
"""
@inline function prepare(nodes::AbstractMatrix{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    return prepare(svectors(nodes), kernel)
end
@inline function prepare(nodes::AbstractVecOfSVecs{n, T}, kernel::RK = RK_H0()) where {n, T <: Real, RK <: ReproducingKernel_0}
    return _prepare(nodes, kernel)
end

"""
`construct(spline::AbstractNormalSpline{n,T,RK}, values::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Construct the spline by calculating its coefficients and completely initializing the `NormalSpline` object.

# Arguments

  - `spline`: the partly initialized `NormalSpline` object returned by `prepare` function.
  - `values`: function values at `nodes` nodes.

# Returns

  - The completely initialized `NormalSpline` object that can be passed to `evaluate` function to interpolate the data to required points.
"""
@inline function construct(spline::AbstractNormalSpline{n, T, RK}, values::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return _construct!(deepcopy(spline), values)
end

"""
`interpolate(nodes::AbstractMatrix{T}, values::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Prepare and construct the spline.

# Arguments

  - `nodes`: The function value nodes.
    This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
    and `n₁` is the number of function value nodes.
    It means that each column in the matrix defines one node.
  - `values`: function values at `nodes` nodes.
  - `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
    It must be a struct object of the following type:
    `RK_H0` if the spline is constructing as a continuous function,
    `RK_H1` if the spline is constructing as a differentiable function,
    `RK_H2` if the spline is constructing as a twice differentiable function.

# Returns

  - The completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
@inline function interpolate(nodes::AbstractMatrix{T}, values::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    return interpolate(svectors(nodes), values, kernel)
end
@inline function interpolate(nodes::AbstractVecOfSVecs{n, T}, values::AbstractVector{T}, kernel::RK = RK_H0()) where {n, T <: Real, RK <: ReproducingKernel_0}
    spline = _prepare(nodes, kernel)
    return _construct!(spline, values)
end

"""
`evaluate(spline::AbstractNormalSpline{n,T,RK}, points::AbstractMatrix{T}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Evaluate the spline values at the locations defined in `points`.

# Arguments

  - `spline: the `NormalSpline`object returned by`interpolate`or`construct` function.
  - `points`: locations at which spline values are evaluating.
    This should be an `n×m` matrix, where `n` is dimension of the sampled space
    and `m` is the number of locations where spline values are evaluating.
    It means that each column in the matrix defines one location.

# Returns

  - `Vector{T}` of the spline values at the locations defined in `points`.
"""
@inline function evaluate(spline::AbstractNormalSpline{n, T1, RK}, points::AbstractMatrix{T2}) where {n, T1 <: Real, T2 <: Real, RK <: ReproducingKernel_0}
    return evaluate(spline, svectors(points))
end
@inline function evaluate(spline::AbstractNormalSpline{n, T1, RK}, points::AbstractArrOfSVecs{n, T2}) where {n, T1 <: Real, T2 <: Real, RK <: ReproducingKernel_0}
    T = promote_type(T1, eltype(RK), T2)
    return evaluate!(zeros(T, size(points)), spline, points)
end
@inline function evaluate!(spline_values::AbstractArray{<:Any, D}, spline::AbstractNormalSpline{n, T1, RK}, points::AbstractArrOfSVecs{n, T2, D}) where {n, D, T1 <: Real, T2 <: Real, RK <: ReproducingKernel_0}
    return _evaluate!(spline_values, spline, points)
end
@inline function evaluate(spline::AbstractNormalSpline{n, T1, RK}, point::SVector{n, T2}) where {n, T1 <: Real, T2 <: Real, RK <: ReproducingKernel_0}
    return _evaluate(spline, point)
end

"""
`evaluate_one(spline::AbstractNormalSpline{n,T,RK}, point::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Evaluate the spline value at the `point` location.

# Arguments

  - `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
  - `point`: location at which spline value is evaluating.
    This should be a vector of size `n`, where `n` is dimension of the sampled space.

# Returns

  - The spline value at the location defined in `point`.
"""
@inline function evaluate_one(spline::AbstractNormalSpline{n, T1, RK}, point::AbstractVector{T2}) where {n, T1 <: Real, T2 <: Real, RK <: ReproducingKernel_0}
    T = promote_type(T1, eltype(RK), T2)
    return evaluate_one(spline, SVector{n, T}(ntuple(i -> point[i], n)))
end
@inline function evaluate_one(spline::AbstractNormalSpline{n, T1, RK}, point::SVector{n, T2}) where {n, T1 <: Real, T2 <: Real, RK <: ReproducingKernel_0}
    return evaluate(spline, point)
end

"""
`evaluate_gradient(spline::AbstractNormalSpline{n,T,RK}, point::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Evaluate gradient of the spline at the location defined in `point`.

# Arguments

  - `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
  - `point`: location at which gradient value is evaluating.
    This should be a vector of size `n`, where `n` is dimension of the sampled space.

Note: Gradient of spline built with reproducing kernel RK_H0 does not exist at the spline nodes.

# Returns

  - `Vector{T}` - gradient of the spline at the location defined in `point`.
"""
@inline function evaluate_gradient(spline::AbstractNormalSpline{n, T1, RK}, point::AbstractVector{T2}) where {n, T1 <: Real, T2 <: Real, RK <: ReproducingKernel_0}
    T = promote_type(T1, eltype(RK), T2)
    return evaluate_gradient(spline, SVector{n, T}(ntuple(i -> point[i], n)))
end
@inline function evaluate_gradient(spline::AbstractNormalSpline{n, T1, RK}, point::SVector{n, T2}) where {n, T1 <: Real, T2 <: Real, RK <: ReproducingKernel_0}
    return _evaluate_gradient(spline, point)
end

#### ReproducingKernel_1

"""
`prepare(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, d_dirs::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Prepare the spline by constructing and factoring a Gram matrix of the interpolation problem.
Initialize the `NormalSpline` object.

# Arguments

  - `nodes`: The function value nodes.
    This should be an `n×n_1` matrix, where `n` is dimension of the sampled space and
    `n₁` is the number of function value nodes.
    It means that each column in the matrix defines one node.
  - `d_nodes`: The function directional derivatives nodes.
    This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
    `n₂` is the number of function directional derivative nodes.
  - `d_dirs`: Directions of the function directional derivatives.
    This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
    `n₂` is the number of function directional derivative nodes.
    It means that each column in the matrix defines one direction of the function directional derivative.
  - `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
    It must be a struct object of the following type:
    `RK_H1` if the spline is constructing as a differentiable function,
    `RK_H2` if the spline is constructing as a twice differentiable function.

# Returns

  - The partly initialized `NormalSpline` object that must be passed to `construct` function in order to complete the spline initialization.
"""
@inline function prepare(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, d_dirs::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    return prepare(svectors(nodes), svectors(d_nodes), svectors(d_dirs), kernel)
end
@inline function prepare(nodes::AbstractVecOfSVecs{n, T}, d_nodes::AbstractVecOfSVecs{n, T}, d_dirs::AbstractVecOfSVecs{n, T}, kernel::RK = RK_H1()) where {n, T <: Real, RK <: ReproducingKernel_1}
    return _prepare(nodes, d_nodes, d_dirs, kernel)
end

"""
`construct(spline::AbstractNormalSpline{n,T,RK}, values::AbstractVector{T}, d_values::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_1}`

Construct the spline by calculating its coefficients and completely initializing the `NormalSpline` object.

# Arguments

  - `spline`: the partly initialized `NormalSpline` object returned by `prepare` function.
  - `values`: function values at `nodes` nodes.
  - `d_values`: function directional derivative values at `d_nodes` nodes.

# Returns

  - The completely initialized `NormalSpline` object that can be passed to `evaluate` function to interpolate the data to required points.
"""
@inline function construct(spline::AbstractNormalSpline{n, T, RK}, values::AbstractVector{T}, d_values::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_1}
    return _construct!(deepcopy(spline), values, d_values)
end

"""
`interpolate(nodes::AbstractMatrix{T}, values::AbstractVector{T}, d_nodes::AbstractMatrix{T}, d_dirs::AbstractMatrix{T}, d_values::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Prepare and construct the spline.

# Arguments

  - `nodes`: The function value nodes.
    This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
    and `n₁` is the number of function value nodes.
    It means that each column in the matrix defines one node.
  - `values`: function values at `nodes` nodes.
  - `d_nodes`: The function directional derivative nodes.
    This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
    `n₂` is the number of function directional derivative nodes.
  - `d_dirs`: Directions of the function directional derivatives.
    This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
    `n₂` is the number of function directional derivative nodes.
    It means that each column in the matrix defines one direction of the function directional derivative.
  - `d_values`: function directional derivative values at `d_nodes` nodes.
  - `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
    It must be a struct object of the following type:
    `RK_H1` if the spline is constructing as a differentiable function,
    `RK_H2` if the spline is constructing as a twice differentiable function.

# Returns

  - The completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
@inline function interpolate(nodes::AbstractMatrix{T}, values::AbstractVector{T}, d_nodes::AbstractMatrix{T}, d_dirs::AbstractMatrix{T}, d_values::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    return interpolate(svectors(nodes), values, svectors(d_nodes), svectors(d_dirs), d_values, kernel)
end
@inline function interpolate(nodes::AbstractVecOfSVecs{n, T}, values::AbstractVector{T}, d_nodes::AbstractVecOfSVecs{n, T}, d_dirs::AbstractVecOfSVecs{n, T}, d_values::AbstractVector{T}, kernel::RK = RK_H1()) where {n, T <: Real, RK <: ReproducingKernel_1}
    spline = _prepare(nodes, d_nodes, d_dirs, kernel)
    return _construct!(spline, values, d_values)
end

#### ReproducingKernel_0 (1-dimensional case)

"""
`prepare(nodes::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Prepare the 1D spline by constructing and factoring a Gram matrix of the interpolation problem.
Initialize the `NormalSpline` object.

# Arguments

  - `nodes`: function value interpolation nodes.
    This should be an `n₁` vector where `n₁` is the number of function value nodes.
  - `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
    It must be a struct object of the following type:
    `RK_H0` if the spline is constructing as a continuous function,
    `RK_H1` if the spline is constructing as a differentiable function,
    `RK_H2` if the spline is constructing as a twice differentiable function.

# Returns

  - The partly initialized `NormalSpline` object that must be passed to `construct` function in order to complete the spline initialization.
"""
@inline function prepare(nodes::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    return prepare(svectors(nodes), kernel)
end

"""
`interpolate(nodes::AbstractVector{T}, values::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Prepare and construct the 1D spline.

# Arguments

  - `nodes`: function value interpolation nodes.
    This should be an `n₁` vector where `n₁` is the number of function value nodes.
  - `values`: function values at `n₁` interpolation nodes.
  - `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
    It must be a struct object of the following type:
    `RK_H0` if the spline is constructing as a continuous function,
    `RK_H1` if the spline is constructing as a differentiable function,
    `RK_H2` if the spline is constructing as a twice differentiable function.

# Returns

  - The completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
@inline function interpolate(nodes::AbstractVector{T}, values::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    return interpolate(svectors(nodes), values, kernel)
end

"""
`evaluate(spline::AbstractNormalSpline{n,T,RK}, points::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Evaluate the 1D spline values/value at the `points` locations.

# Arguments

  - `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
  - `points`: locations at which spline values are evaluating.
    This should be a vector of size `m` where `m` is the number of evaluating points.

# Returns

  - Spline value at the `point` location.
"""
@inline function evaluate(spline::AbstractNormalSpline{n, T1, RK}, points::AbstractVector{T2}) where {n, T1 <: Real, T2 <: Real, RK <: ReproducingKernel_0}
    return evaluate(spline, svectors(points))
end

"""
`evaluate(spline::AbstractNormalSpline{1,T,RK}, point::T) where {T <: Real, RK <: ReproducingKernel_0}`

Evaluate the 1D spline value at the `point` location.

# Arguments

  - `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
  - `point`: location at which spline value is evaluating.

# Returns

  - Spline value at the `point` location.
"""
@inline function evaluate(spline::AbstractNormalSpline{1, T1, RK}, point::T2) where {T1 <: Real, T2 <: Real, RK <: ReproducingKernel_0}
    T = promote_type(T1, eltype(RK), T2)
    return evaluate(spline, SVector{1, T}((point,)))
end

"""
`evaluate_derivative(spline::AbstractNormalSpline{1,T,RK}, point::T) where {T <: Real, RK <: ReproducingKernel_0}`

Evaluate the 1D spline derivative at the `point` location.

# Arguments

  - `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
  - `point`: location at which spline derivative is evaluating.

Note: Derivative of spline built with reproducing kernel RK_H0 does not exist at the spline nodes.

# Returns

  - The spline derivative value at the `point` location.
"""
@inline function evaluate_derivative(spline::AbstractNormalSpline{1, T1, RK}, point::T2) where {T1 <: Real, T2 <: Real, RK <: ReproducingKernel_0}
    T = promote_type(T1, eltype(RK), T2)
    return evaluate_gradient(spline, SVector{1, T}((point,)))[1]
end

#### ReproducingKernel_1 (1-dimensional case)

"""
`prepare(nodes::AbstractVector{T}, d_nodes::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Prepare the 1D interpolating normal spline by constructing and factoring a Gram matrix of the problem.
Initialize the `NormalSpline` object.

# Arguments

  - `nodes`: function value interpolation nodes.
    This should be an `n₁` vector where `n₁` is the number of function value nodes.
  - `d_nodes`: The function derivatives nodes.
    This should be an `n₂` vector where `n₂` is the number of function derivatives nodes.
  - `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
    It must be a struct object of the following type:
    `RK_H1` if the spline is constructing as a differentiable function,
    `RK_H2` if the spline is constructing as a twice differentiable function.

# Returns

  - The partly initialized `NormalSpline` object that must be passed to `construct` function in order to complete the spline initialization.
"""
@inline function prepare(nodes::AbstractVector{T}, d_nodes::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    d_dirs = fill(ones(SVector{1, T}), length(d_nodes))
    return prepare(svectors(nodes), svectors(d_nodes), d_dirs, kernel)
end

"""
`interpolate(nodes::AbstractVector{T}, values::AbstractVector{T}, d_nodes::AbstractVector{T}, d_values::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Prepare and construct the 1D interpolating normal spline.

# Arguments

  - `nodes`: function value interpolation nodes.
    This should be an `n₁` vector where `n₁` is the number of function value nodes.
  - `values`: function values at `nodes` nodes.
  - `d_nodes`: The function derivatives nodes.
    This should be an `n₂` vector where `n₂` is the number of function derivatives nodes.
  - `d_values`: function derivative values at `d_nodes` nodes.
  - `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
    It must be a struct object of the following type:
    `RK_H1` if the spline is constructing as a differentiable function,
    `RK_H2` if the spline is constructing as a twice differentiable function.

# Returns

  - The completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
@inline function interpolate(nodes::AbstractVector{T}, values::AbstractVector{T}, d_nodes::AbstractVector{T}, d_values::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    d_dirs = fill(ones(SVector{1, T}), length(d_nodes))
    return interpolate(svectors(nodes), values, svectors(d_nodes), d_dirs, d_values, kernel)
end

#### Utils for general case

"""
`get_epsilon(spline::AbstractNormalSpline{n,T,RK}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Get the 'scaling parameter' of Bessel Potential space the spline was built in.

# Arguments

  - `spline`: the `NormalSpline` object returned by `prepare`, `construct` or `interpolate` function.

# Returns

  - The 'scaling parameter' `ε`.
"""
@inline function get_epsilon(spline::AbstractNormalSpline{n, T, RK}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return _get_kernel(spline).ε
end

"""
`estimate_epsilon(nodes::AbstractMatrix{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Get the estimation of the 'scaling parameter' of Bessel Potential space the spline being built in.
It coincides with the result returned by `get_epsilon` function.

# Arguments

  - `nodes`: The function value nodes.
    This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
    and `n₁` is the number of function value nodes.
    It means that each column in the matrix defines one node.
  - `kernel`: reproducing kernel of Bessel potential space the normal spline will be constructed in.
    It must be a struct object of the following type:
    `RK_H0` if the spline is constructing as a continuous function,
    `RK_H1` if the spline is constructing as a differentiable function,
    `RK_H2` if the spline is constructing as a twice differentiable function.

# Returns

  - Estimation of `ε`.
"""
@inline function estimate_epsilon(nodes::AbstractMatrix{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    return estimate_epsilon(svectors(nodes), kernel)
end
@inline function estimate_epsilon(nodes::AbstractVecOfSVecs{n, T}, kernel::RK = RK_H0()) where {n, T <: Real, RK <: ReproducingKernel_0}
    return _estimate_epsilon(nodes, kernel)
end

"""
`estimate_epsilon(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Get an the estimation of the 'scaling parameter' of Bessel Potential space the spline being built in.
It coincides with the result returned by `get_epsilon` function.

# Arguments

  - `nodes`: The function value nodes.
    This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
    and `n₁` is the number of function value nodes.
    It means that each column in the matrix defines one node.
  - `d_nodes`: The function directional derivative nodes.
    This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
    `n₂` is the number of function directional derivative nodes.
  - `kernel`: reproducing kernel of Bessel potential space the normal spline will be constructed in.
    It must be a struct object of the following type:
    `RK_H1` if the spline is constructing as a differentiable function,
    `RK_H2` if the spline is constructing as a twice differentiable function.

# Returns

  - Estimation of `ε`.
"""
@inline function estimate_epsilon(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    return estimate_epsilon(svectors(nodes), svectors(d_nodes), kernel)
end
@inline function estimate_epsilon(nodes::AbstractVecOfSVecs{n, T}, d_nodes::AbstractVecOfSVecs{n, T}, kernel::RK = RK_H1()) where {n, T <: Real, RK <: ReproducingKernel_1}
    return _estimate_epsilon(nodes, d_nodes, kernel)
end

"""
`estimate_cond(spline::AbstractNormalSpline{n,T,RK}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Get an estimation of the Gram matrix condition number. It needs the `spline` object is prepared and requires O(N^2) operations.
(C. Brás, W. Hager, J. Júdice, An investigation of feasible descent algorithms for estimating the condition number of a matrix. TOP Vol.20, No.3, 2012.)

# Arguments

  - `spline`: the `NormalSpline` object returned by `prepare`, `construct` or `interpolate` function.

# Returns

  - An estimation of the Gram matrix condition number.
"""
@inline function estimate_cond(spline::AbstractNormalSpline{n, T, RK}) where {n, T <: Real, RK <: ReproducingKernel}
    return _get_cond(spline)
end

"""
`estimate_accuracy(spline::AbstractNormalSpline{n,T,RK}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Assess accuracy of interpolation results by analyzing residuals.

# Arguments

  - `spline`: the `NormalSpline` object returned by `construct` or `interpolate` function.

# Returns

  - An estimation of the number of significant digits in the interpolation result.
"""
@inline function estimate_accuracy(spline::AbstractNormalSpline{n, T, RK}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return _estimate_accuracy(spline)
end

"""
`get_cond(nodes::AbstractMatrix{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Get a value of the Gram matrix spectral condition number. It is obtained by means of the matrix SVD decomposition and requires ``O(N^3)`` operations.

# Arguments

  - `nodes`: The function value nodes.
    This should be an `n×n_1` matrix, where `n` is dimension of the sampled space and
    `n₁` is the number of function value nodes. It means that each column in the matrix defines one node.
  - `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
    It must be a struct object of the following type:
    `RK_H0` if the spline is constructing as a continuous function,
    `RK_H1` if the spline is constructing as a differentiable function,
    `RK_H2` if the spline is constructing as a twice differentiable function.

# Returns

  - A value of the Gram matrix spectral condition number.
"""
@inline function get_cond(nodes::AbstractMatrix{T}, kernel::RK) where {T <: Real, RK <: ReproducingKernel_0}
    return get_cond(svectors(nodes), kernel)
end
@inline function get_cond(nodes::AbstractVecOfSVecs{n, T}, kernel::RK) where {n, T <: Real, RK <: ReproducingKernel_0}
    return get_cond(nodes, kernel)
end

"""
`get_cond(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, d_dirs::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Get a value of the Gram matrix spectral condition number. It is obtained by means of the matrix SVD decomposition and requires ``O(N^3)`` operations.

# Arguments

  - `nodes`: The function value nodes.
    This should be an `n×n_1` matrix, where `n` is dimension of the sampled space and
    `n₁` is the number of function value nodes.
    It means that each column in the matrix defines one node.
  - `d_nodes`: The function directional derivatives nodes.
    This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
    `n₂` is the number of function directional derivative nodes.
  - `d_dirs`: Directions of the function directional derivatives.
    This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
    `n₂` is the number of function directional derivative nodes.
    It means that each column in the matrix defines one direction of the function directional derivative.
  - `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
    It must be a struct object of the following type:
    `RK_H1` if the spline is constructing as a differentiable function,
    `RK_H2` if the spline is constructing as a twice differentiable function.

# Returns

  - A value of the Gram matrix spectral condition number.
"""
@inline function get_cond(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, d_dirs::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    return get_cond(svectors(nodes), svectors(d_nodes), svectors(d_dirs), kernel)
end
@inline function get_cond(nodes::AbstractVecOfSVecs{n, T}, d_nodes::AbstractVecOfSVecs{n, T}, d_dirs::AbstractVecOfSVecs{n, T}, kernel::RK = RK_H1()) where {n, T <: Real, RK <: ReproducingKernel_1}
    return get_cond(nodes, d_nodes, d_dirs, kernel)
end

#### Utils for 1-dimensional case

"""
`estimate_epsilon(nodes::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Get an the estimation of the 'scaling parameter' of Bessel Potential space the 1D spline is being built in.
It coincides with the result returned by `get_epsilon` function.

# Arguments

  - `nodes`: The function value nodes.
  - `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
    It must be a struct object of the following type:
    `RK_H0` if the spline is constructing as a continuous function,
    `RK_H1` if the spline is constructing as a differentiable function,
    `RK_H2` if the spline is constructing as a twice differentiable function.

# Returns

  - Estimation of `ε`.
"""
@inline function estimate_epsilon(nodes::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    return estimate_epsilon(svectors(nodes), kernel)
end

"""
`estimate_epsilon(nodes::AbstractVector{T}, d_nodes::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Get an the estimation of the 'scaling parameter' of Bessel Potential space the 1D spline is being built in.
It coincides with the result returned by `get_epsilon` function.

# Arguments

  - `nodes`: The function value nodes.
  - `d_nodes`: The function derivative nodes.
  - `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
    It must be a struct object of the following type:
    `RK_H1` if the spline is constructing as a differentiable function,
    `RK_H2` if the spline is constructing as a twice differentiable function.

# Returns

  - Estimation of `ε`.
"""
@inline function estimate_epsilon(nodes::AbstractVector{T}, d_nodes::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    return estimate_epsilon(svectors(nodes), svectors(d_nodes), kernel)
end

end # module NormalHermiteSplines
