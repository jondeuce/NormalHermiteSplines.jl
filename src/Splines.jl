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
Base.@kwdef struct NormalSpline{n, T <: Real, RK <: ReproducingKernel_0} <: AbstractNormalSpline{n,T,RK}
    _kernel::RK
    _nodes::VecOfSVecs{n,T}
    _values::Vector{T}        = zeros(T, 0)
    _d_nodes::VecOfSVecs{n,T} = zeros(SVector{n,T}, 0)
    _d_dirs::VecOfSVecs{n,T}  = zeros(SVector{n,T}, 0)
    _d_values::Vector{T}      = zeros(T, 0)
    _mu::Vector{T}            = zeros(T, 0)
    _gram::Hermitian{T, Matrix{T}}
    _chol::Cholesky{T, Matrix{T}}
    _cond::T
    _min_bound::SVector{n,T}
    _max_bound::SVector{n,T}
    _scale::T
end

Base.@kwdef struct ElasticNormalSpline{n, T <: Real, RK <: ReproducingKernel_0} <: AbstractNormalSpline{n,T,RK}
    _kernel::RK
    _max_size::Int
    _num_nodes::Base.RefValue{Int}      = Ref(0)
    _num_d_nodes::Base.RefValue{Int}    = Ref(0)
    _nodes::VecOfSVecs{n,T}             = zeros(SVector{n,T}, _max_size)
    _values::Vector{T}                  = zeros(T, _max_size)
    _d_nodes::VecOfSVecs{n,T}           = zeros(SVector{n,T}, n * _max_size)
    _d_dirs::VecOfSVecs{n,T}            = zeros(SVector{n,T}, n * _max_size)
    _d_values::Vector{T}                = zeros(T, n * _max_size)
    _mu::Vector{T}                      = zeros(T, (n+1) * _max_size)
    _chol::ElasticCholesky{T,Matrix{T}} = ElasticCholesky{T}((n+1) * _max_size)
    _min_bound::SVector{n,T}
    _max_bound::SVector{n,T}
    _scale::T
end
@inline _get_nodes(spl::ElasticNormalSpline)     = view(spl._nodes, 1:spl._num_nodes[])
@inline _get_values(spl::ElasticNormalSpline)    = view(spl._values, 1:spl._num_nodes[])
@inline _get_d_nodes(spl::ElasticNormalSpline)   = view(spl._d_nodes, 1:spl._num_d_nodes[])
@inline _get_d_dirs(spl::ElasticNormalSpline)    = view(spl._d_dirs, 1:spl._num_d_nodes[])
@inline _get_d_values(spl::ElasticNormalSpline)  = view(spl._d_values, 1:spl._num_d_nodes[])
@inline _get_mu(spl::ElasticNormalSpline)        = view(spl._mu, 1:spl._num_nodes[]+spl._num_d_nodes[])
@inline _get_cond(spl::ElasticNormalSpline)      = _estimate_cond(_get_gram(spl), _get_chol(spl))
@inline function _get_gram(spl::ElasticNormalSpline{<:Any, <:Any, <:ReproducingKernel_0})
    n₁ = spl._num_nodes[]
    return Hermitian(view(Base.parent(spl._chol), 1:n₁, 1:n₁), :U)
end

function ElasticNormalSpline(min_bound::SVector{n,T}, max_bound::SVector{n,T}, max_size::Int, kernel::RK) where {n, T, RK <: ReproducingKernel_0}
    @assert kernel.ε != 0
    scale = maximum(max_bound .- min_bound)
    ElasticNormalSpline{n,T,RK}(; _kernel = kernel, _max_size = max_size, _min_bound = min_bound, _max_bound = max_bound, _scale = scale)
end

function Base.empty!(spl::ElasticNormalSpline)
    spl._num_nodes[] = 0
    spl._num_d_nodes[] = 0
    empty!(spl._chol)
    return spl
end

function Base.insert!(
        spl::ElasticNormalSpline{n,T,RK},
        node::SVector{n,T},
        value::T,
    ) where {n, T, RK <: ReproducingKernel_0}

    n₁ = spl._num_nodes[]
    @assert n₁ < spl._max_size
    # @assert n₂ < n * spl._max_size

    # Normalize + insert node (assumed to be with `min_bound` and `_max_bound`)
    new_node = _normalize(spl, node)
    @inbounds spl._nodes[n₁+1] = new_node
    @inbounds spl._values[n₁+1] = value

    # Update Gram matrix + Cholesky factorization
    curr_nodes = uview(spl._nodes, 1:n₁)
    gram = uview(Base.parent(spl._chol), 1:n₁+1, 1:n₁+1)
    _gram!(gram, new_node, curr_nodes, spl._kernel)
    cholesky!(spl._chol, n₁+1)

    # Compute spline coefficients
    μ = uview(spl._mu, 1:n₁+1)
    v = uview(spl._values, 1:n₁+1)
    ldiv!(μ, spl._chol, v)

    spl._num_nodes[] += 1

    return nothing
end
