function _normalization_scaling(nodes::AbstractVecOfSVecs)
    min_bound = reduce((x, y) -> min.(x, y), nodes)
    max_bound = reduce((x, y) -> max.(x, y), nodes)
    compression = maximum(max_bound .- min_bound)
    return min_bound, compression
end

function _normalization_scaling(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs)
    min_bound = min.(reduce((x, y) -> min.(x, y), nodes), reduce((x, y) -> min.(x, y), d_nodes))
    max_bound = max.(reduce((x, y) -> max.(x, y), nodes), reduce((x, y) -> max.(x, y), d_nodes))
    compression = maximum(max_bound .- min_bound)
    return min_bound, compression
end

function _estimate_accuracy(spline::NormalSpline{n,T,RK}) where {n,T,RK <: ReproducingKernel_0}
    nodes = (x -> spline._min_bound .+ spline._compression .* x).(spline._nodes)
    σ = _evaluate!(zeros(T, length(nodes)), spline, nodes)
    # calculating a value of the Relative Maximum Absolute Error (RMAE) of interpolation
    # at the function value interpolation nodes.
    fun_max = maximum(abs, spline._values)
    if fun_max > 0
        rmae = maximum(abs.(spline._values .- σ)) / fun_max
    else
        rmae = maximum(abs.(spline._values .- σ))
    end
    rmae = rmae > eps(T) ? rmae : eps(T)
    #= 
    # calculating a value of the Relative Maximum Absolute Error (RMAE) of interpolation
    # at the function value interpolation nodes.
    fun_max = maximum(abs, spline._values)
    for i in 1:length(spline._nodes_normalized)
        node = spline._min_bound .+ spline._compression .* spline._nodes_normalized[i]
        σ = _evaluate!(zeros(T, length(nodes)), spline, nodes)
        if fun_max > 0
            rmae = maximum(abs.(spline._values .- σ)) / fun_max
        else
            rmae = maximum(abs.(spline._values .- σ))
        end
        rmae = rmae > eps(T) ? rmae : eps(T)
    end =#
    res = -floor(log10(rmae)) - 1
    if res <= 0
        res = 0
    end
    return trunc(Int, res)
end

function _pairwise_sum_norms(nodes::AbstractVecOfSVecs{n,T}) where {n,T}
    ℓ = zero(T)
    @inbounds for i in 1:length(nodes), j in i:length(nodes)
        ℓ += norm(nodes[i] .- nodes[j])
    end
    return ℓ
end

function _pairwise_sum_norms_weighted(nodes::AbstractVecOfSVecs{n,T}, d_nodes::AbstractVecOfSVecs{n,T}, w_d_nodes::T) where {n,T}
    ℓ = zero(T)
    @inbounds for i in 1:length(nodes), j in 1:length(d_nodes)
        ℓ += norm(nodes[i] .- w_d_nodes .* d_nodes[j])
    end
    return ℓ
end

function _estimate_ε(nodes::AbstractVecOfSVecs{n,T}) where {n,T}
    n_1 = length(nodes)
    ε = _pairwise_sum_norms(nodes)
    return ε > 0 ? ε * T(n)^T(inv(n)) / T(n_1)^(T(5) / 3) : one(T)
end

        function _estimate_ε(nodes::AbstractVecOfSVecs{n,T}, d_nodes::AbstractVecOfSVecs{n,T}, w_d_nodes::T=T(0.1)) where {n,T}
    n_1 = length(nodes)
    n_2 = length(d_nodes)
    ε = _pairwise_sum_norms(nodes) + _pairwise_sum_norms_weighted(nodes, d_nodes, w_d_nodes) + w_d_nodes * _pairwise_sum_norms(d_nodes)
    return ε > 0 ? ε * T(n)^T(inv(n)) / T(n_1 + n_2)^(T(5) / 3) : one(T)
end

function _estimate_epsilon(nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    min_bound, compression = _normalization_scaling(nodes)
    t_nodes = (x -> (x .- min_bound) ./ compression).(nodes)
    ε = _estimate_ε(t_nodes)
    if isa(kernel, RK_H1)
        ε *= 3 // 2
    elseif isa(kernel, RK_H2)
        ε *= 2
    end
    return ε
end

function _estimate_epsilon(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_1)
    min_bound, compression = _normalization_scaling(nodes, d_nodes)
    t_nodes = (x -> (x .- min_bound) ./ compression).(nodes)
    t_d_nodes = (x -> (x .- min_bound) ./ compression).(d_nodes)

    ε = _estimate_ε(t_nodes, t_d_nodes)
    if isa(kernel, RK_H1)
        ε *= 2
    elseif isa(kernel, RK_H2)
        ε *= 5 // 2
    end

    return ε
end

function _get_gram(nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    min_bound, compression = _normalization_scaling(nodes)
    t_nodes = (x -> (x .- min_bound) ./ compression).(nodes)

    if kernel.ε == 0
        ε = _estimate_ε(t_nodes)
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

    return _gram!(zeros(T, n_1, n_1), t_nodes, kernel)
end

function _get_gram(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs, es::AbstractVecOfSVecs, kernel::ReproducingKernel_1)
    min_bound, compression = _normalization_scaling(nodes, d_nodes)
    t_nodes = (x -> (x .- min_bound) ./ compression).(nodes)
    t_d_nodes = (x -> (x .- min_bound) ./ compression).(d_nodes)
    t_es = es ./ norm.(es)

    if kernel.ε == 0
        ε = _estimate_ε(t_nodes, t_d_nodes)
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

    return _gram!(zeros(T, n_1 + n_2, n_1 + n_2), t_nodes, t_d_nodes, t_es, kernel)
end

function _get_cond(nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    T = promote_type(typeof(kernel.ε), eltype(eltype(nodes)))
    gram = _get_gram(nodes, kernel)
    cond = zero(T)
    try
        evs = svdvals!(gram)
        maxevs = maximum(evs)
        minevs = minimum(evs)
        if minevs > 0
            cond = maxevs / minevs
            cond = T(10)^floor(log10(cond))
        end
    catch
    end
    return cond
end
    
function _get_cond(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs, es::AbstractVecOfSVecs, kernel::ReproducingKernel_1)
    _get_cond(nodes, kernel)
end

# ```
# Get estimation of the Gram matrix condition number
# Brás, C.P., Hager, W.W. & Júdice, J.J. An investigation of feasible descent algorithms for estimating the condition number of a matrix. TOP 20, 791–809 (2012).
# https://link.springer.com/article/10.1007/s11750-010-0161-9
# ```
function _estimate_cond(
        gram::AbstractMatrix{T},
        chol::LinearAlgebra.Cholesky{T,Array{T,2}},
        nit=3
    ) where {T}
    mat_norm = norm(gram, 1)
    n = size(gram, 1)
    x = fill(inv(T(n)), n)
    z = Vector{T}(undef, n)
    gamma = zero(T)
    for it = 1:nit
        z = ldiv!(z, chol, x)
        gamma = zero(T)
        for i = 1:n
            gamma += abs(z[i])
            z[i] = sign(z[i])
        end
        z = ldiv!(z, chol, copy(z))
        zx = z ⋅ x
        idx = 1
        for i = 1:n
            z[i] = abs(z[i])
            if z[i] > z[idx]
                idx = i
            end
        end
        if z[idx] <= zx
            break
        end
        x .= 0
        x[idx] = 1
    end
    cond = T(10.0)^floor(log10(mat_norm * gamma))
    return cond
end
