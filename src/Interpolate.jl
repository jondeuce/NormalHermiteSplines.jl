#### Construct Normal spline (ReproducingKernel_0)

function _prepare(nodes::AbstractVecOfSVecs{n, T}, kernel::ReproducingKernel_0) where {n, T}
    # Normalize nodes out-of-place to avoid aliasing
    min_bound, max_bound, scale = _normalization_scaling(nodes)
    nodes = _normalize.(nodes, (min_bound,), (max_bound,), scale)

    if kernel.ε == 0
        kernel = _estimate_ε(kernel, nodes)
    end

    gram = _gram(nodes, kernel)
    chol = cholesky(gram)
    cond = _estimate_cond(gram, chol)

    return NormalSpline{n, T, typeof(kernel)}(; _kernel = kernel, _nodes = nodes, _gram = gram, _chol = chol, _cond = cond, _min_bound = min_bound, _max_bound = max_bound, _scale = scale)
end

function _construct!(
    spl::NormalSpline{n, T, RK},
    values::AbstractVector{T},
) where {n, T, RK <: ReproducingKernel_0}
    n₁ = length(values)
    @assert length(_get_nodes(spl)) == n₁ "Number of data values ($n₁) does not correspond to the number of nodes $(length(_get_nodes(spl)))."
    @assert size(_get_chol(spl)) == (n₁, n₁) "Number of data values ($n₁) does not correspond to the size of the Gram matrix ($(size(_get_chol(spl))))."

    # Resize buffers
    resize!(_get_values(spl), n₁)
    empty!(_get_d_nodes(spl))
    empty!(_get_d_dirs(spl))
    empty!(_get_d_values(spl))
    resize!(_get_mu(spl), n₁)
    resize!(_get_rhs(spl), n₁)

    # Copy values to avoid aliasing
    _get_values(spl) .= _get_rhs(spl) .= values

    # Compute spline coefficients
    ldiv!(_get_mu(spl), _get_chol(spl), _get_rhs(spl))

    return spl
end

#### Construct Normal spline (ReproducingKernel_1)

function _prepare(nodes::AbstractVecOfSVecs{n, T}, d_nodes::AbstractVecOfSVecs{n, T}, d_dirs::AbstractVecOfSVecs{n, T}, kernel::ReproducingKernel_1) where {n, T}
    # Normalize inputs out-of-place to avoid aliasing
    min_bound, max_bound, scale = _normalization_scaling(nodes, d_nodes)
    nodes = _normalize.(nodes, (min_bound,), (max_bound,), scale)
    d_nodes = _normalize.(d_nodes, (min_bound,), (max_bound,), scale)
    d_dirs = d_dirs ./ _norm.(d_dirs)

    if kernel.ε == 0
        kernel = _estimate_ε(kernel, nodes, d_nodes)
    end

    gram = _gram(nodes, d_nodes, d_dirs, kernel)
    chol = cholesky(gram)
    cond = _estimate_cond(gram, chol)

    return NormalSpline{n, T, typeof(kernel)}(; _kernel = kernel, _nodes = nodes, _d_nodes = d_nodes, _d_dirs = d_dirs, _gram = gram, _chol = chol, _cond = cond, _min_bound = min_bound, _max_bound = max_bound, _scale = scale)
end

function _construct!(
    spl::NormalSpline{n, T, RK},
    values::AbstractVector{T},
    d_values::AbstractVector{T},
) where {n, T, RK <: ReproducingKernel_1}
    n₁ = length(values)
    n₂ = length(d_values)
    @assert length(_get_nodes(spl)) == n₁ "Number of data values ($n₁) does not correspond to the number of nodes $(length(_get_nodes(spl)))."
    @assert length(_get_d_nodes(spl)) == n₂ "Number of derivative values ($n₂) does not correspond to the number of derivative nodes."
    @assert size(_get_chol(spl)) == (n₁ + n₂, n₁ + n₂) "Number of data and derivative values ($(n₁+n₂)) do not correspond to the size of the Gram matrix ($(size(_get_chol(spl))))."

    # Resize buffers
    resize!(_get_values(spl), n₁)
    resize!(_get_d_nodes(spl), n₂)
    resize!(_get_d_dirs(spl), n₂)
    resize!(_get_d_values(spl), n₂)
    resize!(_get_mu(spl), n₁ + n₂)
    resize!(_get_rhs(spl), n₁ + n₂)

    # Copy values to avoid aliasing
    _get_values(spl) .= view(_get_rhs(spl), 1:n₁) .= values

    # Nodes scaled down by `scale` -> directional derivative scaled up by `scale`; allocate new array to avoid aliasing
    _get_d_values(spl) .= view(_get_rhs(spl), n₁+1:n₁+n₂) .= _get_scale(spl) .* d_values

    # Compute spline coefficients and construct spline
    ldiv!(_get_mu(spl), _get_chol(spl), _get_rhs(spl))

    return spl
end

#### Evaluation

function _evaluate!(
    vals::AbstractArray{<:Any, N},
    spl::AbstractNormalSpline{n, <:Any, <:ReproducingKernel_0},
    points::AbstractArrOfSVecs{n, <:Any, N},
) where {n, N}
    @inbounds for i in eachindex(vals, points)
        vals[i] = _evaluate(spl, points[i])
    end
    return vals
end

function _evaluate(
    spl::AbstractNormalSpline{n, <:Any, RK},
    x::SVector{n},
) where {n, RK <: ReproducingKernel_0}
    kernel, nodes, d_nodes, d_dirs, mu =
        _get_kernel(spl), _get_nodes(spl), _get_d_nodes(spl), _get_d_dirs(spl), _get_mu(spl)
    n₁ = length(nodes)
    n₂ = length(d_nodes)
    x = _normalize(spl, x)
    T = promote_type(eltype(spl), eltype(kernel), eltype(x))
    v = zero(T)
    @inbounds for i in 1:n₁
        v += mu[i] * _rk(kernel, x, nodes[i])
    end
    if RK <: ReproducingKernel_1
        @inbounds for i in 1:n₂
            v -= mu[i+n₁] * _∂rk_∂ηⁱ_ûᵢ(kernel, x, d_nodes[i], d_dirs[i])
        end
    end
    return v
end

function _evaluate_gradient(
    spl::AbstractNormalSpline{n, <:Any, RK},
    x::SVector{n},
) where {n, RK <: ReproducingKernel_0}
    kernel, nodes, d_nodes, d_dirs, mu, scale =
        _get_kernel(spl), _get_nodes(spl), _get_d_nodes(spl), _get_d_dirs(spl), _get_mu(spl), _get_scale(spl)
    n₁ = length(nodes)
    n₂ = length(d_nodes)
    x = _normalize(spl, x)
    T = promote_type(eltype(spl), eltype(kernel), eltype(x))
    ∇ = zero(SVector{n, T})
    @inbounds for i in 1:n₁
        ∇ += mu[i] * _∂rk_∂η(kernel, x, nodes[i])
    end
    if RK <: ReproducingKernel_1
        @inbounds for i in 1:n₂
            ∇ += mu[i+n₁] * _∂²rk_∂ηⁱ∂ξ_ûᵢ(kernel, x, d_nodes[i], d_dirs[i])
        end
    end
    return ∇ ./ scale
end

function _evaluate_with_gradient(
    spl::AbstractNormalSpline{n, <:Any, RK},
    x::SVector{n},
) where {n, RK <: ReproducingKernel_0}
    kernel, nodes, d_nodes, d_dirs, mu, scale =
        _get_kernel(spl), _get_nodes(spl), _get_d_nodes(spl), _get_d_dirs(spl), _get_mu(spl), _get_scale(spl)
    n₁ = length(nodes)
    n₂ = length(d_nodes)
    x = _normalize(spl, x)
    T = promote_type(eltype(spl), eltype(kernel), eltype(x))
    v = zero(T)
    ∇ = zero(SVector{n, T})
    @inbounds for i in 1:n₁
        μ = mu[i]
        vᵢ, ∇ᵢ = _rk_with_∂rk_∂η(kernel, x, nodes[i])
        v += μ * vᵢ
        ∇ += μ * ∇ᵢ
    end
    if RK <: ReproducingKernel_1
        @inbounds for i in 1:n₂
            μ = mu[i+n₁]
            vᵢ, ∇ᵢ = _∂rk_∂ηⁱ_ûᵢ_with_∂²rk_∂ηⁱ∂ξ_ûᵢ(kernel, x, d_nodes[i], d_dirs[i])
            v -= μ * vᵢ
            ∇ += μ * ∇ᵢ
        end
    end
    return v, ∇ ./ scale
end
