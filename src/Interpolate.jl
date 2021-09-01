function _prepare(nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    min_bound, compression = _normalization_scaling(nodes)
    nodes_normalized = (x -> (x .- min_bound) ./ compression).(nodes)

    if kernel.ε == 0
        ε = _estimate_ε(nodes_normalized)
        if isa(kernel, RK_H0)
            kernel = RK_H0(ε)
        elseif isa(kernel, RK_H1)
            ε *= 3 // 2
            kernel = RK_H1(ε)
        elseif isa(kernel, RK_H2)
            ε *= 2
            kernel = RK_H2(ε)
        else
            error("incorrect `kernel` type.")
        end
    end

    T = eltype(eltype(nodes_normalized))
    n_1 = length(nodes)
    gram = _gram!(zeros(T, n_1, n_1), nodes_normalized, kernel)
    chol = nothing
    try
        chol = cholesky(gram)
    catch
        error("Cannot prepare the spline: Gram matrix is degenerate.")
    end

    cond = _estimate_cond(gram, chol)

    spline = NormalSpline(kernel,
                          compression,
                          copy(nodes),
                          nodes_normalized,
                          nothing,
                          nothing,
                          nothing,
                          nothing,
                          nothing,
                          nothing,
                          min_bound,
                          gram,
                          chol,
                          nothing,
                          cond,
                          )
    return spline
end

function _construct(spline::NormalSpline{n,T,RK},
                    values::AbstractVector{T},
                    cleanup::Bool=false
                   ) where {n,T,RK <: ReproducingKernel_0}
    if length(values) != length(spline._nodes)
        error("Number of data values does not correspond to the number of nodes.")
    end
    if isnothing(spline._chol)
        error("Gram matrix was not factorized.")
    end

    mu = Vector{T}(undef, size(spline._gram, 1))
    ldiv!(mu, spline._chol, values)

    spline = NormalSpline(spline._kernel,
                          spline._compression,
                          spline._nodes,
                          spline._nodes_normalized,
                          values,
                          nothing,
                          nothing,
                          nothing,
                          nothing,
                          nothing,
                          spline._min_bound,
                          cleanup ? nothing : spline._gram,
                          cleanup ? nothing : spline._chol,
                          mu,
                          spline._cond,
                          )
    return spline
end

###################

function _prepare(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs, es::AbstractVecOfSVecs, kernel::ReproducingKernel_1)
    min_bound, compression = _normalization_scaling(nodes, d_nodes)
    nodes_normalized = (x -> (x .- min_bound) ./ compression).(nodes)
    d_nodes_normalized = (x -> (x .- min_bound) ./ compression).(d_nodes)
    es_normalized = es ./ norm.(es)

    if kernel.ε == 0
        ε = _estimate_ε(nodes_normalized, d_nodes_normalized)
        if isa(kernel, RK_H1)
            ε *= 2
            kernel = RK_H1(ε)
        elseif isa(kernel, RK_H2)
            ε *= 5 // 2
            kernel = RK_H2(ε)
        else
            error("incorrect `kernel` type.")
        end
    end

    n_1 = length(nodes)
    n_2 = length(d_nodes)
    T = promote_type(eltype(eltype(nodes_normalized)), eltype(eltype(d_nodes_normalized)), eltype(eltype(es_normalized)))
    gram = _gram!(zeros(T, n_1 + n_2, n_1 + n_2), nodes_normalized, d_nodes_normalized, es_normalized, kernel)
    chol = nothing
    try
        chol = cholesky(gram)
    catch
        error("Cannot prepare the spline: Gram matrix is degenerate.")
    end

    cond = _estimate_cond(gram, chol)

    spline = NormalSpline(kernel,
                          compression,
                          copy(nodes),
                          nodes_normalized,
                          nothing,
                          copy(d_nodes),
                          d_nodes_normalized,
                          copy(es),
                          es_normalized,
                          nothing,
                          min_bound,
                          gram,
                          chol,
                          nothing,
                          cond,
                          )
    return spline
end

function _construct(spline::NormalSpline{n,T,RK},
                    values::AbstractVector{T},
                    d_values::AbstractVector{T},
                    cleanup::Bool=false
                   ) where {n,T,RK <: ReproducingKernel_0}
    if length(values) != length(spline._nodes)
        error("Number of data values does not correspond to the number of nodes.")
    end
    if length(d_values) != length(spline._d_nodes)
        error("Number of derivative values does not correspond to the number of derivative nodes.")
    end
    if isnothing(spline._chol)
        error("Gram matrix was not factorized.")
    end

    mu = Vector{T}(undef, size(spline._gram, 1))
    ldiv!(mu, spline._chol, [values; spline._compression .* d_values])

    spline = NormalSpline(spline._kernel,
                          spline._compression,
                          spline._nodes,
                          spline._nodes_normalized,
                          values,
                          spline._d_nodes,
                          spline._d_nodes_normalized,
                          spline._es,
                          spline._es_normalized,
                          d_values,
                          spline._min_bound,
                          cleanup ? nothing : spline._gram,
                          cleanup ? nothing : spline._chol,
                          mu,
                          spline._cond,
                          )
    return spline
end

function _evaluate!(
        spline_values::AbstractVector,
        spline::NormalSpline{n, <:Any, <:ReproducingKernel_0},
        points::AbstractVecOfSVecs{n},
    ) where {n}
    for i = 1:length(points)
        spline_values[i] = _evaluate(spline, points[i])
    end
    return spline_values
end

function _evaluate(
        spline::NormalSpline{n, <:Any, <:ReproducingKernel_0},
        point::SVector{n},
    ) where {n}
    n_1 = length(spline._nodes_normalized)
    mu = @views spline._mu[1:n_1]
    d_mu = @views spline._mu[n_1 + 1:end]
    point = (point .- spline._min_bound) ./ spline._compression

    if isnothing(spline._d_nodes_normalized)
        _evaluate(point, spline._nodes_normalized, mu, spline._kernel)
    else
        _evaluate(point, spline._nodes_normalized, spline._d_nodes_normalized, spline._es_normalized, mu, d_mu, spline._kernel)
    end
end

@inline function _evaluate(
        point::SVector,
        nodes::AbstractVecOfSVecs,
        mu::AbstractVector,
        kernel::ReproducingKernel_0,
    )
    T = promote_type(typeof(kernel.ε), eltype(point), eltype(eltype(nodes)), eltype(mu))
    v = zero(T)
    @inbounds for i in 1:length(nodes)
        v += mu[i] * _rk(kernel, point, nodes[i])
    end
    return v
end

@inline function _evaluate(
        point::SVector,
        nodes::AbstractVecOfSVecs,
        d_nodes::AbstractVecOfSVecs,
        es::AbstractVecOfSVecs,
        mu::AbstractVector,
        d_mu::AbstractVector,
        kernel::ReproducingKernel_1,
    )
    v = _evaluate(point, nodes, mu, kernel)
    for i in 1:length(d_nodes)
        v += d_mu[i] * _∂rk_∂e(kernel, point, d_nodes[i], es[i])
    end
    return v
end

function _evaluate_gradient(
        spline::NormalSpline{n, <:Any, <:ReproducingKernel_0},
        point::SVector{n},
    ) where {n}
    n_1 = length(spline._nodes_normalized)
    mu = @views spline._mu[1:n_1]
    d_mu = @views spline._mu[n_1 + 1:end]
    point = (point .- spline._min_bound) ./ spline._compression

    if isnothing(spline._d_nodes_normalized)
        _evaluate_gradient(point, spline._nodes_normalized, mu, spline._kernel) ./ spline._compression
    else
        _evaluate_gradient(point, spline._nodes_normalized, spline._d_nodes_normalized, spline._es_normalized, mu, d_mu, spline._kernel) ./ spline._compression
    end
end

@inline function _evaluate_gradient(
        point::SVector{n},
        nodes::AbstractVecOfSVecs{n},
        mu::AbstractVector,
        kernel::ReproducingKernel_0,
    ) where {n}
    T = promote_type(typeof(kernel.ε), eltype(point), eltype(eltype(nodes)), eltype(mu))
    grad = zero(SVector{n,T})
    @inbounds for i = 1:length(nodes)
        μ = mu[i]
        node = nodes[i]
        grad += SVector(ntuple(k -> μ * _∂rk_∂η_k(kernel, point, node, k), n))
    end
    return grad
end

@inline function _evaluate_gradient(
        point::SVector{n},
        nodes::AbstractVecOfSVecs{n},
        d_nodes::AbstractVecOfSVecs{n},
        es::AbstractVecOfSVecs{n},
        mu::AbstractVector,
        d_mu::AbstractVector,
        kernel::ReproducingKernel_1,
    ) where {n}
    grad = _evaluate_gradient(point, nodes, mu, kernel)
    RK = CartesianIndices((n, n))
    @inbounds for i = 1:length(d_nodes)
        ∂μ = d_mu[i]
        ∂node = d_nodes[i]
        ê = es[i]
        ∂ = ntuple(n * n) do j
            r, k = Tuple(RK[j])
            _∂²rk_∂η_r_∂ξ_k(kernel, point, ∂node, r, k)
        end
        ∂ = SMatrix{n,n}(∂)
        grad += ∂μ * (∂ * ê)
    end
    return grad
end
