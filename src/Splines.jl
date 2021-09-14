abstract type AbstractNormalSpline{n,T,RK} end

Base.ndims(::AbstractNormalSpline{n,T,RK}) where {n,T,RK} = n
Base.eltype(::AbstractNormalSpline{n,T,RK}) where {n,T,RK} = T

@inline _get_kernel(spl::AbstractNormalSpline)    = spl._kernel
@inline _get_nodes(spl::AbstractNormalSpline)     = spl._nodes
@inline _get_values(spl::AbstractNormalSpline)    = spl._values
@inline _get_d_nodes(spl::AbstractNormalSpline)   = spl._d_nodes
@inline _get_d_dirs(spl::AbstractNormalSpline)    = spl._d_dirs
@inline _get_d_values(spl::AbstractNormalSpline)  = spl._d_values
@inline _get_mu(spl::AbstractNormalSpline)        = spl._mu
@inline _get_gram(spl::AbstractNormalSpline)      = spl._gram
@inline _get_chol(spl::AbstractNormalSpline)      = spl._chol
@inline _get_cond(spl::AbstractNormalSpline)      = spl._cond
@inline _get_min_bound(spl::AbstractNormalSpline) = spl._min_bound
@inline _get_max_bound(spl::AbstractNormalSpline) = spl._max_bound
@inline _get_scale(spl::AbstractNormalSpline)     = spl._scale

@doc raw"
`struct NormalSpline{n, T <: Real, RK <: ReproducingKernel_0} <: AbstractNormalSpline{n,T,RK}`

Define a structure containing full information of a normal spline
# Fields
- `_kernel`: a reproducing kernel spline was built with
- `_scale`: factor of transforming the original node locations into unit hypercube
- `_nodes`: transformed function value nodes
- `_values`: function values at interpolation nodes
- `_d_nodes`: transformed function directional derivative nodes
- `_d_dirs`: normalized derivative directions
- `_d_values`: function directional derivative values
- `_min_bound`: minimal bounds of the original node locations area
- `_gram`: Gram matrix of the problem
- `_chol`: Cholesky factorization of the Gram matrix
- `_mu`: spline coefficients
- `_cond`: estimation of the Gram matrix condition number
"
struct NormalSpline{n, T <: Real, RK <: ReproducingKernel_0} <: AbstractNormalSpline{n,T,RK}
    _kernel::RK
    _nodes::VecOfSVecs{n,T}
    _values::Vector{T}
    _d_nodes::VecOfSVecs{n,T}
    _d_dirs::VecOfSVecs{n,T}
    _d_values::Vector{T}
    _mu::Vector{T}
    _gram::Hermitian{T, Matrix{T}}
    _chol::Cholesky{T, Matrix{T}}
    _cond::T
    _min_bound::SVector{n,T}
    _max_bound::SVector{n,T}
    _scale::T
end
