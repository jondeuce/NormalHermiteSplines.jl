abstract type AbstractNormalSpline{n, T, RK} end

Base.ndims(::AbstractNormalSpline{n, T, RK}) where {n, T, RK} = n
Base.eltype(::AbstractNormalSpline{n, T, RK}) where {n, T, RK} = T

@inline _get_kernel(spl::AbstractNormalSpline)    = spl._kernel
@inline _get_nodes(spl::AbstractNormalSpline)     = spl._nodes
@inline _get_values(spl::AbstractNormalSpline)    = spl._values
@inline _get_d_nodes(spl::AbstractNormalSpline)   = spl._d_nodes
@inline _get_d_dirs(spl::AbstractNormalSpline)    = spl._d_dirs
@inline _get_d_values(spl::AbstractNormalSpline)  = spl._d_values
@inline _get_mu(spl::AbstractNormalSpline)        = spl._mu
@inline _get_rhs(spl::AbstractNormalSpline)       = spl._rhs
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
- `_nodes`: transformed function value nodes
- `_values`: function values at interpolation nodes
- `_d_nodes`: transformed function directional derivative nodes
- `_d_dirs`: normalized derivative directions
- `_d_values`: function directional derivative values
- `_mu`: spline coefficients
- `_rhs`: right-hand side of the problem `gram * mu = rhs`
- `_gram`: Gram matrix of the problem `gram * mu = rhs`
- `_chol`: Cholesky factorization of the Gram matrix
- `_cond`: estimation of the Gram matrix condition number
- `_min_bound`: minimal bounds of the original node locations area
- `_max_bound`: maximal bounds of the original node locations area
- `_scale`: factor of transforming the original node locations into unit hypercube
"
Base.@kwdef struct NormalSpline{n, T <: Real, RK <: ReproducingKernel_0} <: AbstractNormalSpline{n, T, RK}
    _kernel                    :: RK
    _nodes                     :: VecOfSVecs{n, T}
    _values::Vector{T}         = zeros(T, 0)
    _d_nodes::VecOfSVecs{n, T} = zeros(SVector{n, T}, 0)
    _d_dirs::VecOfSVecs{n, T}  = zeros(SVector{n, T}, 0)
    _d_values::Vector{T}       = zeros(T, 0)
    _mu::Vector{T}             = zeros(T, 0)
    _rhs::Vector{T}            = zeros(T, 0)
    _gram                      :: Hermitian{T, Matrix{T}}
    _chol                      :: Cholesky{T, Matrix{T}}
    _cond                      :: T
    _min_bound                 :: SVector{n, T}
    _max_bound                 :: SVector{n, T}
    _scale                     :: T
end

Base.@kwdef struct ElasticNormalSpline{n, T <: Real, RK <: ReproducingKernel_0} <: AbstractNormalSpline{n, T, RK}
    _kernel                              :: RK
    _max_size                            :: Int
    _num_nodes::Base.RefValue{Int}       = Ref(0)
    _num_d_nodes::Base.RefValue{Int}     = Ref(0)
    _nodes::VecOfSVecs{n, T}             = zeros(SVector{n, T}, _max_size)
    _values::Vector{T}                   = zeros(T, _max_size)
    _d_nodes::VecOfSVecs{n, T}           = zeros(SVector{n, T}, n * _max_size)
    _d_dirs::VecOfSVecs{n, T}            = zeros(SVector{n, T}, n * _max_size)
    _d_values::Vector{T}                 = zeros(T, n * _max_size)
    _mu::Vector{T}                       = zeros(T, (n + 1) * _max_size)
    _rhs::Vector{T}                      = zeros(T, (n + 1) * _max_size)
    _gram::Matrix{T}                     = zeros(T, (n + 1) * _max_size, (n + 1) * _max_size)
    _chol::ElasticCholesky{T, Matrix{T}} = ElasticCholesky{T}((n + 1) * _max_size)
    _filled_columns::Vector{Int}         = zeros(Int, (n + 1) * _max_size)
    _min_bound                           :: SVector{n, T}
    _max_bound                           :: SVector{n, T}
    _scale                               :: T
end
function ElasticNormalSpline(min_bound::SVector{n, T}, max_bound::SVector{n, T}, max_size::Int, kernel::RK) where {n, T, RK <: ReproducingKernel_0}
    @assert kernel.ε != 0
    scale = maximum(max_bound .- min_bound)
    return ElasticNormalSpline{n, T, RK}(; _kernel = kernel, _max_size = max_size, _min_bound = min_bound, _max_bound = max_bound, _scale = scale)
end

@inline _get_nodes(spl::ElasticNormalSpline)           = uview(spl._nodes, 1:spl._num_nodes[])
@inline _get_values(spl::ElasticNormalSpline)          = uview(spl._values, 1:spl._num_nodes[])
@inline _get_d_nodes(spl::ElasticNormalSpline)         = uview(spl._d_nodes, 1:spl._num_d_nodes[])
@inline _get_d_dirs(spl::ElasticNormalSpline)          = uview(spl._d_dirs, 1:spl._num_d_nodes[])
@inline _get_d_values(spl::ElasticNormalSpline)        = uview(spl._d_values, 1:spl._num_d_nodes[])
@inline _get_cond(spl::ElasticNormalSpline)            = _estimate_cond(_get_gram(spl), _get_chol(spl))
@inline _get_mu(spl::ElasticNormalSpline)              = (J = _get_filled_columns(spl); return uview(spl._mu, J))
@inline _get_rhs(spl::ElasticNormalSpline)             = (J = _get_filled_columns(spl); return uview(spl._rhs, J))
@inline _get_gram(spl::ElasticNormalSpline)            = (J = _get_filled_columns(spl); A = uview(spl._gram, J, J); return Hermitian(A, :U))
@inline _get_filled_columns(spl::ElasticNormalSpline)  = uview(spl._filled_columns, 1:spl._num_nodes[]+spl._num_d_nodes[])
@inline _get_insertion_order(spl::ElasticNormalSpline) = uview(_get_chol(spl).colperms, 1:_get_chol(spl).ncols[])

function insertat!(x::AbstractVector, i, v, len = length(x))
    last = v
    @inbounds for j in i:len
        x[j], last = last, x[j]
    end
    return x
end

function Base.empty!(spl::ElasticNormalSpline)
    spl._num_nodes[] = 0
    spl._num_d_nodes[] = 0
    empty!(spl._chol)
    return spl
end

function Base.insert!(
    spl::ElasticNormalSpline{n, T, RK},
    node::SVector{n, T},
    value::T,
) where {n, T, RK <: ReproducingKernel_0}

    n₁, n₂ = spl._num_nodes[], spl._num_d_nodes[]
    n₁max  = spl._max_size
    n₂max  = n * spl._max_size
    @assert n₁ < n₁max

    # Normalize and insert node (assumed to be with `min_bound` and `_max_bound`)
    curr_nodes   = _get_nodes(spl)
    curr_d_nodes = _get_d_nodes(spl)
    curr_d_dirs  = _get_d_dirs(spl)
    new_node     = _normalize(spl, node)
    new_value    = value
    @inbounds begin
        spl._nodes[n₁+1]  = new_node
        spl._values[n₁+1] = new_value
        spl._rhs[n₁+1]    = new_value
        insertat!(spl._filled_columns, n₁ + 1, n₁ + 1, n₁ + n₂ + 1)
        spl._num_nodes[] += 1
    end

    # Insert column into position `n₁+1` of Gram matrix
    inds = _get_filled_columns(spl)
    if RK <: ReproducingKernel_1
        _gram!(uview(spl._gram, inds, inds), new_node, curr_nodes, curr_d_nodes, curr_d_dirs, _get_kernel(spl))
    else
        _gram!(uview(spl._gram, inds, inds), new_node, curr_nodes, _get_kernel(spl))
    end

    # Insert column `n₁+1` of Gram matrix into Cholesky factorization
    insert!(spl._chol, n₁ + 1, Hermitian(spl._gram))
    cholesky!(spl._chol, n₁ + 1)

    # Solve for spline coefficients
    inds = _get_insertion_order(spl)
    ldiv!(uview(spl._mu, inds), spl._chol, uview(spl._rhs, inds))

    return nothing
end

function Base.insert!(
    spl::ElasticNormalSpline{n, T, RK},
    d_node::SVector{n},
    d_dir::SVector{n},
    d_value::T,
) where {n, T, RK <: ReproducingKernel_1}

    n₁, n₂ = spl._num_nodes[], spl._num_d_nodes[]
    n₁max  = spl._max_size
    n₂max  = n * spl._max_size
    @assert n₂ < n₂max

    # Normalize and insert node (assumed to be with `min_bound` and `_max_bound`)
    curr_nodes   = _get_nodes(spl)
    curr_d_nodes = _get_d_nodes(spl)
    curr_d_dirs  = _get_d_dirs(spl)
    new_d_node   = _normalize(spl, d_node)
    new_d_dir    = d_dir / _norm(d_dir)
    new_d_value  = _get_scale(spl) * d_value
    @inbounds begin
        spl._d_nodes[n₂+1]   = new_d_node
        spl._d_values[n₂+1]  = new_d_value
        spl._d_dirs[n₂+1]    = new_d_dir
        spl._rhs[n₁max+n₂+1] = new_d_value
        insertat!(spl._filled_columns, n₁ + n₂ + 1, n₁max + n₂ + 1, n₁ + n₂ + 1)
        spl._num_d_nodes[] += 1
    end

    # Insert column into position `n₁max+n₂+1` of Gram matrix
    inds = _get_filled_columns(spl)
    _gram!(uview(spl._gram, inds, inds), new_d_node, new_d_dir, curr_nodes, curr_d_nodes, curr_d_dirs, _get_kernel(spl))

    # Insert column `n₁max+n₂+1` of Gram matrix into Cholesky factorization
    insert!(spl._chol, n₁max + n₂ + 1, Hermitian(spl._gram))
    cholesky!(spl._chol, n₁max + n₂ + 1)

    # Solve for spline coefficients
    inds = _get_insertion_order(spl)
    ldiv!(uview(spl._mu, inds), spl._chol, uview(spl._rhs, inds))

    return nothing
end

function Base.insert!(
    spl::ElasticNormalSpline{n, T, RK},
    nodes::AbstractVecOfSVecs{n, T},
    values::AbstractVector{T},
) where {n, T, RK <: ReproducingKernel_0}
    @assert length(nodes) == length(values)

    # Insert `n` regular nodes
    @inbounds for i in 1:length(nodes)
        insert!(spl, nodes[i], values[i])
    end

    return spl
end

function Base.insert!(
    spl::ElasticNormalSpline{n, T, RK},
    nodes::AbstractVecOfSVecs{n, T},
    values::AbstractVector{T},
    d_nodes::AbstractVecOfSVecs{n, T},
    d_dirs::AbstractVecOfSVecs{n, T},
    d_values::AbstractVector{T},
) where {n, T, RK <: ReproducingKernel_1}
    @assert length(nodes) == length(values)
    @assert length(d_nodes) == length(d_dirs) == length(d_values)

    # Insert `n` regular nodes
    @inbounds for i in 1:length(nodes)
        insert!(spl, nodes[i], values[i])
    end

    # Insert `n` derivative nodes
    @inbounds for i in 1:length(d_nodes)
        insert!(spl, d_nodes[i], d_dirs[i], d_values[i])
    end

    return spl
end
