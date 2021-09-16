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

struct ElasticNormalSpline{n, T <: Real, RK <: ReproducingKernel_0} <: AbstractNormalSpline{n,T,RK}
    _kernel::Base.RefValue{RK}
    _max_size::Int
    _num_nodes::Base.RefValue{Int}
    _num_d_nodes::Base.RefValue{Int}
    _nodes::VecOfSVecs{n,T}
    _values::Vector{T}
    _d_nodes::VecOfSVecs{n,T}
    _d_dirs::VecOfSVecs{n,T}
    _d_values::Vector{T}
    _mu::Vector{T}
    _chol::ElasticCholesky{T}
    _min_bound::Base.RefValue{SVector{n,T}}
    _max_bound::Base.RefValue{SVector{n,T}}
    _scale::Base.RefValue{T}
end
@inline _get_kernel(spl::ElasticNormalSpline)    = spl._kernel[]
@inline _get_nodes(spl::ElasticNormalSpline)     = uview(spl._nodes, 1:spl._num_nodes[])
@inline _get_values(spl::ElasticNormalSpline)    = uview(spl._values, 1:spl._num_nodes[])
@inline _get_d_nodes(spl::ElasticNormalSpline)   = uview(spl._d_nodes. 1:spl._num_d_nodes[])
@inline _get_d_dirs(spl::ElasticNormalSpline)    = uview(spl._d_dirs. 1:spl._num_d_nodes[])
@inline _get_d_values(spl::ElasticNormalSpline)  = uview(spl._d_values. 1:spl._num_d_nodes[])
@inline _get_mu(spl::ElasticNormalSpline)        = uview(spl._mu, 1:spl._num_nodes[]+spl._num_d_nodes[])
@inline _get_gram(spl::ElasticNormalSpline)      = Base.parent(spl._chol)
@inline _get_chol(spl::ElasticNormalSpline)      = spl._chol
@inline _get_cond(spl::ElasticNormalSpline)      = _estimate_cond(_get_gram(spl), _get_chol(spl))
@inline _get_min_bound(spl::ElasticNormalSpline) = spl._min_bound[]
@inline _get_max_bound(spl::ElasticNormalSpline) = spl._max_bound[]
@inline _get_scale(spl::ElasticNormalSpline)     = spl._scale[]

function ElasticNormalSpline{n, T}(kernel::RK; maxsize::Int) where {n, T, RK <: ReproducingKernel_0}
    @show SVector{n,T}
    _num_nodes = Ref(0)
    _num_d_nodes = Ref(0)
    _nodes = zeros(SVector{n,T}, maxsize)
    _values = zeros(T, maxsize)
    _d_nodes = zeros(SVector{n,T}, n * maxsize)
    _d_dirs = zeros(SVector{n,T}, n * maxsize)
    _d_values = zeros(T, n * maxsize)
    _mu = zeros(T, (n+1) * maxsize)
    _chol = ElasticCholesky{T}(; maxsize = (n+1) * maxsize)
    _min_bound = Ref(fill(zero(T), SVector{n,T}))
    _max_bound = Ref(fill(one(T), SVector{n,T}))
    _scale = Ref(one(T))
    ElasticNormalSpline{n, T, RK}(
        Ref(kernel),
        maxsize,
        _num_nodes,
        _num_d_nodes,
        _nodes,
        _values,
        _d_nodes,
        _d_dirs,
        _d_values,
        _mu,
        _chol,
        _min_bound,
        _max_bound,
        _scale,
    )
end

function _insert!(
        spl::ElasticNormalSpline{n,T,RK},
        node::SVector{n,T},
        value::T,
    ) where {n, T, RK <: ReproducingKernel_0}

    # TODO: Normalize nodes
    @assert spl._num_nodes[] <= spl._max_size
    @assert spl._num_d_nodes[] <= n * spl._max_size
    spl._nodes[spl._num_nodes[] + 1] = node
    spl._values[spl._num_nodes[] + 1] = value
    spl._num_nodes[] += 1
end

function _factorize!(
        spl::ElasticNormalSpline{n,T,RK},
    ) where {n, T, RK <: ReproducingKernel_0}

    # TODO: Renormalize nodes
    # min_bound, max_bound, scale = _normalization_scaling(_get_nodes(spl))
    # if scale > 1
    #     spl._min_bound[] = min.(get_min_bound(spl), min_bound .* scale)
    #     spl._max_bound[] = max.(get_max_bound(spl), max_bound .* scale)
    #     spl._scale[] *= scale
    #     _get_nodes(spl) .= _normalize.(_get_nodes(spl), (min_bound,), (max_bound,), scale)
    # end

    # Update kernel and compute Gram matrix
    spl._kernel[] = _estimate_ε(spl._kernel[], _get_nodes(spl))
    _gram!(_get_gram(spl), _get_nodes(spl), _get_kernel(spl))
    for j in 1:spl._num_nodes[]
        _factorize_column!(_get_chol(spl), j)
    end

    # Compute spline coefficients
    ldiv!(_get_mu(spl), _get_chol(spl), _get_values(spl))

    return spl
end

function test_elastic_spline()
    n, T = 1, Float64
    maxsize = 5
    spl = ElasticNormalSpline{n,T}(RK_H0(); maxsize = maxsize);
    for _ in 1:4
        _insert!(spl, rand(SVector{n,T}), rand(T))
    end
    _factorize!(spl)
    return spl
end
