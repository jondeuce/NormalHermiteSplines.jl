module NormalHermiteSplines

#### Inteface deinition
export prepare, construct, interpolate
export evaluate, evaluate_one, evaluate_gradient
export NormalSpline, RK_H0, RK_H1, RK_H2
export get_epsilon, estimate_epsilon, get_cond, estimate_cond
export estimate_accuracy
export evaluate_derivative

using LinearAlgebra
using StaticArrays

abstract type ReproducingKernel end
abstract type ReproducingKernel_0 <: ReproducingKernel end
abstract type ReproducingKernel_1 <: ReproducingKernel_0 end
abstract type ReproducingKernel_2 <: ReproducingKernel_1 end

abstract type AbstractSpline end

const AbstractVecOfSVecs{n,T} = AbstractVector{SVector{n,T}}
const VecOfSVecs{n,T} = Vector{SVector{n,T}}

@inline svectors(x::AbstractMatrix{T}) where {T} = reinterpret(reshape, SVector{size(x,1),T}, x)
@inline svectors(x::AbstractVector{T}) where {T} = reinterpret(SVector{1,T}, x)

@doc raw"
`struct NormalSpline{n, T, RK} <: AbstractSpline where {n, T <: Real, RK <: ReproducingKernel_0}`

Define a structure containing full information of a normal spline
# Fields
- `_kernel`: a reproducing kernel spline was built with
- `_compression`: factor of transforming the original node locations into unit hypercube
- `_nodes`: transformed function value nodes
- `_values`: function values at interpolation nodes
- `_d_nodes`: transformed function directional derivative nodes
- `_es`: normalized derivative directions
- `_d_values`: function directional derivative values
- `_min_bound`: minimal bounds of the original node locations area
- `_gram`: Gram matrix of the problem
- `_chol`: Cholesky factorization of the Gram matrix
- `_mu`: spline coefficients
- `_cond`: estimation of the Gram matrix condition number
"
struct NormalSpline{n, T, RK} <: AbstractSpline where {n, T <: Real, RK <: ReproducingKernel_0}
    _kernel::RK
    _nodes::Union{VecOfSVecs{n,T}, Nothing}
    _values::Union{Vector{T}, Nothing}
    _d_nodes::Union{VecOfSVecs{n,T}, Nothing}
    _es::Union{VecOfSVecs{n,T}, Nothing}
    _d_values::Union{Vector{T}, Nothing}
    _min_bound::SVector{n,T}
    _max_bound::SVector{n,T}
    _compression::T
    _gram::Union{Matrix{T}, Nothing}
    _chol::Union{Cholesky{T, Matrix{T}}, Nothing}
    _mu::Union{Vector{T}, Nothing}
    _cond::T
end

include("./ReproducingKernels.jl")
include("./GramMatrix.jl")
include("./Utils.jl")
include("./Interpolate.jl")

"""
`prepare(nodes::AbstractMatrix{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Prepare the spline by constructing and factoring a Gram matrix of the interpolation problem.
Initialize the `NormalSpline` object.
# Arguments
- `nodes`: The function value nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space and
           `n_1` is the number of function value nodes. It means that each column in the matrix defines one node.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the partly initialized `NormalSpline` object that must be passed to `construct` function
        in order to complete the spline initialization.
"""
@inline function prepare(nodes::AbstractMatrix{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    return prepare(svectors(nodes), kernel)
end
@inline function prepare(nodes::AbstractVecOfSVecs{n,T}, kernel::RK = RK_H0()) where {n, T <: Real, RK <: ReproducingKernel_0}
    spline = _prepare(nodes, kernel)
    return spline
end

"""
`construct(spline::NormalSpline{n,T,RK}, values::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Construct the spline by calculating its coefficients and completely initializing the `NormalSpline` object.
# Arguments
- `spline`: the partly initialized `NormalSpline` object returned by `prepare` function.
- `values`: function values at `nodes` nodes.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function
        to interpolate the data to required points.
"""
@inline function construct(spline::NormalSpline{n,T,RK}, values::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    spline = _construct(spline, values)
    return spline
end

"""
`interpolate(nodes::AbstractMatrix{T}, values::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Prepare and construct the spline.
# Arguments
- `nodes`: The function value nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
           and `n_1` is the number of function value nodes.
           It means that each column in the matrix defines one node.
- `values`: function values at `nodes` nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
@inline function interpolate(nodes::AbstractMatrix{T}, values::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    return interpolate(svectors(nodes), values, kernel)
end
@inline function interpolate(nodes::AbstractVecOfSVecs{n,T}, values::AbstractVector{T}, kernel::RK = RK_H0()) where {n, T <: Real, RK <: ReproducingKernel_0}
    spline = _prepare(nodes, kernel)
    spline = _construct(spline, values)
    return spline
end

"""
`evaluate(spline::NormalSpline{n,T,RK}, points::AbstractMatrix{T}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Evaluate the spline values at the locations defined in `points`.

# Arguments
- `spline: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `points`: locations at which spline values are evaluating.
            This should be an `n×m` matrix, where `n` is dimension of the sampled space
            and `m` is the number of locations where spline values are evaluating.
            It means that each column in the matrix defines one location.

Return: `Vector{T}` of the spline values at the locations defined in `points`.
"""
@inline function evaluate(spline::NormalSpline{n,T,RK}, points::AbstractMatrix{T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return evaluate(spline, svectors(points))
end
@inline function evaluate(spline::NormalSpline{n,T,RK}, points::AbstractVecOfSVecs{n,T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    spline_values = zeros(T, length(points))
    return evaluate!(spline_values, spline, points)
end
@inline function evaluate!(spline_values::AbstractVector{T}, spline::NormalSpline{n,T,RK}, points::AbstractVecOfSVecs{n,T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return _evaluate!(spline_values, spline, points)
end

"""
`evaluate_one(spline::NormalSpline{n,T,RK}, point::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Evaluate the spline value at the `point` location.

# Arguments
- `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `point`: location at which spline value is evaluating.
           This should be a vector of size `n`, where `n` is dimension of the sampled space.

Return: the spline value at the location defined in `point`.
"""
@inline function evaluate_one(spline::NormalSpline{n,T,RK}, point::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return evaluate_one(spline, SVector{n,T}(ntuple(i -> point[i], n)))
end
@inline function evaluate_one(spline::NormalSpline{n,T,RK}, point::SVector{n,T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return _evaluate(spline, point)
end

"""
`evaluate_gradient(spline::NormalSpline{n,T,RK}, point::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Evaluate gradient of the spline at the location defined in `point`.

# Arguments
- `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `point`: location at which gradient value is evaluating.
           This should be a vector of size `n`, where `n` is dimension of the sampled space.

Note: Gradient of spline built with reproducing kernel RK_H0 does not exist at the spline nodes.

Return: `Vector{T}` - gradient of the spline at the location defined in `point`.
"""
@inline function evaluate_gradient(spline::NormalSpline{n,T,RK}, point::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return evaluate_gradient(spline, SVector{n,T}(ntuple(i -> point[i], n)))
end
@inline function evaluate_gradient(spline::NormalSpline{n,T,RK}, point::SVector{n,T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return _evaluate_gradient(spline, point)
end

########

"""
`prepare(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, es::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Prepare the spline by constructing and factoring a Gram matrix of the interpolation problem.
Initialize the `NormalSpline` object.
# Arguments
- `nodes`: The function value nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space and
           `n_1` is the number of function value nodes.
            It means that each column in the matrix defines one node.
- `d_nodes`: The function directional derivatives nodes.
             This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
             `n_2` is the number of function directional derivative nodes.
- `es`: Directions of the function directional derivatives.
        This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
        `n_2` is the number of function directional derivative nodes.
        It means that each column in the matrix defines one direction of the function directional derivative.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the partly initialized `NormalSpline` object that must be passed to `construct` function
        in order to complete the spline initialization.
"""
@inline function prepare(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, es::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    return prepare(svectors(nodes), svectors(d_nodes), svectors(es), kernel)
end
@inline function prepare(nodes::AbstractVecOfSVecs{n,T}, d_nodes::AbstractVecOfSVecs{n,T}, es::AbstractVecOfSVecs{n,T}, kernel::RK = RK_H1()) where {n, T <: Real, RK <: ReproducingKernel_1}
    spline = _prepare(nodes, d_nodes, es, kernel)
    return spline
end

"""
`construct(spline::NormalSpline{n,T,RK}, values::AbstractVector{T}, d_values::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_1}`

Construct the spline by calculating its coefficients and completely initializing the `NormalSpline` object.
# Arguments
- `spline`: the partly initialized `NormalSpline` object returned by `prepare` function.
- `values`: function values at `nodes` nodes.
- `d_values`: function directional derivative values at `d_nodes` nodes.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function
        to interpolate the data to required points.
"""
@inline function construct(spline::NormalSpline{n,T,RK}, values::AbstractVector{T}, d_values::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_1}
    spline = _construct(spline, values, d_values)
    return spline
end

"""
`interpolate(nodes::AbstractMatrix{T}, values::AbstractVector{T}, d_nodes::AbstractMatrix{T}, es::AbstractMatrix{T}, d_values::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Prepare and construct the spline.
# Arguments
- `nodes`: The function value nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
           and `n_1` is the number of function value nodes.
           It means that each column in the matrix defines one node.
- `values`: function values at `nodes` nodes.
- `d_nodes`: The function directional derivative nodes.
            This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
            `n_2` is the number of function directional derivative nodes.
- `es`: Directions of the function directional derivatives.
       This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
       `n_2` is the number of function directional derivative nodes.
       It means that each column in the matrix defines one direction of the function directional derivative.
- `d_values`: function directional derivative values at `d_nodes` nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
@inline function interpolate(nodes::AbstractMatrix{T}, values::AbstractVector{T}, d_nodes::AbstractMatrix{T}, es::AbstractMatrix{T}, d_values::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    return interpolate(svectors(nodes), values, svectors(d_nodes), svectors(es), d_values, kernel)
end
@inline function interpolate(nodes::AbstractVecOfSVecs{n,T}, values::AbstractVector{T}, d_nodes::AbstractVecOfSVecs{n,T}, es::AbstractVecOfSVecs{n,T}, d_values::AbstractVector{T}, kernel::RK = RK_H1()) where {n, T <: Real, RK <: ReproducingKernel_1}
    spline = _prepare(nodes, d_nodes, es, kernel)
    spline = _construct(spline, values, d_values)
    return spline
end

"""
`get_epsilon(spline::NormalSpline{n,T,RK}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Get the 'scaling parameter' of Bessel Potential space the spline was built in.
# Arguments
- `spline`: the `NormalSpline` object returned by `prepare`, `construct` or `interpolate` function.

Return: `ε` - the 'scaling parameter'.
"""
@inline function get_epsilon(spline::NormalSpline{n,T,RK}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return spline._kernel.ε
end

"""
`estimate_epsilon(nodes::AbstractMatrix{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Get the estimation of the 'scaling parameter' of Bessel Potential space the spline being built in.
It coincides with the result returned by `get_epsilon` function.
# Arguments
- `nodes`: The function value nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
           and `n_1` is the number of function value nodes.
           It means that each column in the matrix defines one node.
- `kernel`: reproducing kernel of Bessel potential space the normal spline will be constructed in.
           It must be a struct object of the following type:
             `RK_H0` if the spline is constructing as a continuous function,
             `RK_H1` if the spline is constructing as a differentiable function,
             `RK_H2` if the spline is constructing as a twice differentiable function.
Return: estimation of `ε`.
"""
@inline function estimate_epsilon(nodes::AbstractMatrix{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    return estimate_epsilon(svectors(nodes), kernel)
end
@inline function estimate_epsilon(nodes::AbstractVecOfSVecs{n,T}, kernel::RK = RK_H0()) where {n, T <: Real, RK <: ReproducingKernel_0}
    ε = _estimate_epsilon(nodes, kernel)
    return ε
end

"""
`estimate_epsilon(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Get an the estimation of the 'scaling parameter' of Bessel Potential space the spline being built in.
It coincides with the result returned by `get_epsilon` function.
# Arguments
- `nodes`: The function value nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
           and `n_1` is the number of function value nodes.
           It means that each column in the matrix defines one node.
- `d_nodes`: The function directional derivative nodes.
           This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
           `n_2` is the number of function directional derivative nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline will be constructed in.
            It must be a struct object of the following type:
            `RK_H1` if the spline is constructing as a differentiable function,
            `RK_H2` if the spline is constructing as a twice differentiable function.

Return: estimation of `ε`.
"""
@inline function estimate_epsilon(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    return estimate_epsilon(svectors(nodes), svectors(d_nodes), kernel)
end
@inline function estimate_epsilon(nodes::AbstractVecOfSVecs{n,T}, d_nodes::AbstractVecOfSVecs{n,T}, kernel::RK = RK_H1()) where {n, T <: Real, RK <: ReproducingKernel_1}
    ε = _estimate_epsilon(nodes, d_nodes, kernel)
    return ε
end

"""
`estimate_cond(spline::NormalSpline{n,T,RK}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Get an estimation of the Gram matrix condition number. It needs the `spline` object is prepared and requires O(N^2) operations.
(C. Brás, W. Hager, J. Júdice, An investigation of feasible descent algorithms for estimating the condition number of a matrix. TOP Vol.20, No.3, 2012.)
# Arguments
- `spline`: the `NormalSpline` object returned by `prepare`, `construct` or `interpolate` function.

Return: an estimation of the Gram matrix condition number.
"""
@inline function estimate_cond(spline::NormalSpline{n,T,RK}) where {n, T <: Real, RK <: ReproducingKernel}
    return spline._cond
end

"""
`estimate_accuracy(spline::NormalSpline{n,T,RK}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Assess accuracy of interpolation results by analyzing residuals.
# Arguments
- `spline`: the `NormalSpline` object returned by `construct` or `interpolate` function.

Return: an estimation of the number of significant digits in the interpolation result.
"""
@inline function estimate_accuracy(spline::NormalSpline{n,T,RK}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return _estimate_accuracy(spline)
end

############################## One-dimensional case

"""
`prepare(nodes::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Prepare the 1D spline by constructing and factoring a Gram matrix of the interpolation problem.
Initialize the `NormalSpline` object.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n_1` vector where `n_1` is the number of function value nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the partly initialized `NormalSpline` object that must be passed to `construct` function
        in order to complete the spline initialization.
"""
@inline function prepare(nodes::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    return prepare(svectors(nodes), kernel)
end

"""
`interpolate(nodes::AbstractVector{T}, values::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Prepare and construct the 1D spline.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n_1` vector where `n_1` is the number of function value nodes.
- `values`: function values at `n_1` interpolation nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
@inline function interpolate(nodes::AbstractVector{T}, values::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    interpolate(svectors(nodes), values, kernel)
end

"""
`evaluate(spline::NormalSpline{n,T,RK}, points::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Evaluate the 1D spline values/value at the `points` locations.

# Arguments
- `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `points`: locations at which spline values are evaluating.
            This should be a vector of size `m` where `m` is the number of evaluating points.

Return: spline value at the `point` location.
"""
@inline function evaluate(spline::NormalSpline{n,T,RK}, points::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return evaluate(spline, svectors(points))
end

"""
`evaluate_one(spline::NormalSpline{1,T,RK}, point::T) where {T <: Real, RK <: ReproducingKernel_0}`

Evaluate the 1D spline value at the `point` location.

# Arguments
- `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `point`: location at which spline value is evaluating.

Return: spline value at the `point` location.
"""
@inline function evaluate_one(spline::NormalSpline{1,T,RK}, point::T) where {T <: Real, RK <: ReproducingKernel_0}
    return _evaluate(spline, SVector{1,T}((point,)))
end

"""
`evaluate_derivative(spline::NormalSpline{1,T,RK}, point::T) where {T <: Real, RK <: ReproducingKernel_0}`

Evaluate the 1D spline derivative at the `point` location.

# Arguments
- `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `point`: location at which spline derivative is evaluating.

Note: Derivative of spline built with reproducing kernel RK_H0 does not exist at the spline nodes.

Return: the spline derivative value at the `point` location.
"""
@inline function evaluate_derivative(spline::NormalSpline{1,T,RK}, point::T) where {T <: Real, RK <: ReproducingKernel_0}
    return _evaluate_gradient(spline, SVector{1,T}((point,)))[1]
end

"""
`prepare(nodes::AbstractVector{T}, d_nodes::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Prepare the 1D interpolating normal spline by constructing and factoring a Gram matrix of the problem.
Initialize the `NormalSpline` object.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n_1` vector where `n_1` is the number of function value nodes.
- `d_nodes`: The function derivatives nodes.
             This should be an `n_2` vector where `n_2` is the number of function derivatives nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the partly initialized `NormalSpline` object that must be passed to `construct` function
        in order to complete the spline initialization.
"""
@inline function prepare(nodes::AbstractVector{T}, d_nodes::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    es = fill(ones(SVector{1,T}), length(d_nodes))
    return prepare(svectors(nodes), svectors(d_nodes), es, kernel)
end

"""
`interpolate(nodes::AbstractVector{T}, values::AbstractVector{T}, d_nodes::AbstractVector{T}, d_values::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Prepare and construct the 1D interpolating normal spline.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n_1` vector where `n_1` is the number of function value nodes.
- `values`: function values at `nodes` nodes.
- `d_nodes`: The function derivatives nodes.
             This should be an `n_2` vector where `n_2` is the number of function derivatives nodes.
- `d_values`: function derivative values at `d_nodes` nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
@inline function interpolate(
        nodes::AbstractVector{T},
        values::AbstractVector{T},
        d_nodes::AbstractVector{T},
        d_values::AbstractVector{T},
        kernel::RK = RK_H1()
    ) where {T <: Real, RK <: ReproducingKernel_1}
    es = fill(ones(SVector{1,T}), length(d_nodes))
    return interpolate(svectors(nodes), values, svectors(d_nodes), es, d_values, kernel)
end

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

Return: estimation of `ε`.
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

Return: estimation of `ε`.
"""
@inline function estimate_epsilon(nodes::AbstractVector{T}, d_nodes::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    return estimate_epsilon(svectors(nodes), svectors(d_nodes), kernel)
end

"""
`get_cond(nodes::AbstractMatrix{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Get a value of the Gram matrix spectral condition number. It is obtained by means of the matrix SVD decomposition and requires ``O(N^3)`` operations.
# Arguments
- `nodes`: The function value nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space and
           `n_1` is the number of function value nodes. It means that each column in the matrix defines one node.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: a value of the Gram matrix spectral condition number.
"""
@inline function get_cond(nodes::AbstractMatrix{T}, kernel::RK) where {T <: Real, RK <: ReproducingKernel_0}
    return get_cond(svectors(nodes), kernel)
end
@inline function get_cond(nodes::AbstractVecOfSVecs{n,T}, kernel::RK) where {n, T <: Real, RK <: ReproducingKernel_0}
    return _get_cond(nodes, kernel)
end

"""
`get_cond(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, es::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Get a value of the Gram matrix spectral condition number. It is obtained by means of the matrix SVD decomposition and requires ``O(N^3)`` operations.
# Arguments
- `nodes`: The function value nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space and
           `n_1` is the number of function value nodes.
            It means that each column in the matrix defines one node.
- `d_nodes`: The function directional derivatives nodes.
             This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
             `n_2` is the number of function directional derivative nodes.
- `es`: Directions of the function directional derivatives.
        This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
        `n_2` is the number of function directional derivative nodes.
        It means that each column in the matrix defines one direction of the function directional derivative.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: a value of the Gram matrix spectral condition number.
"""
@inline function get_cond(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, es::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    return get_cond(svectors(nodes), svectors(d_nodes), svectors(es), kernel)
end
@inline function get_cond(nodes::AbstractVecOfSVecs{n,T}, d_nodes::AbstractVecOfSVecs{n,T}, es::AbstractVecOfSVecs{n,T}, kernel::RK = RK_H1()) where {n, T <: Real, RK <: ReproducingKernel_1}
    return _get_cond(nodes, d_nodes, es, kernel)
end

end # module
