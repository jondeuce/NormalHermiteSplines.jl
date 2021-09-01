function _gram!(mat::AbstractMatrix, nodes::AbstractVectorOfSVectors, kernel::ReproducingKernel_0)
    n_1 = length(nodes)
    @inbounds for l = 1:size(mat, 2)
        for i = 1:l
            mat[i,l] = _rk(kernel, nodes[i], nodes[l])
            mat[l,i] = mat[i,l]
        end
    end
    return mat
end
function _gram(nodes::AbstractVectorOfSVectors, kernel::ReproducingKernel_0) where {S <: SVector}
    T = promote_type(typeof(kernel.ε), eltype(eltype(nodes)), )
    n_1 = length(nodes)
    _gram!(zeros(T, n_1, n_1), nodes, kernel)
end
_gram(nodes::AbstractMatrix, kernel::ReproducingKernel_0) = _gram(to_vectors(nodes), kernel)

function _gram!(
        mat::AbstractMatrix,
        nodes::AbstractVectorOfSVectors{n},
        d_nodes::AbstractVectorOfSVectors{n},
        es::AbstractVectorOfSVectors{n},
        kernel::ReproducingKernel_1,
    ) where {n}
    m_1 = length(nodes)
    m_2 = length(d_nodes)
    m = m_1 + m_2
    @inbounds for j = 1:m_1
        for i = 1:j
            mat[i,j] = _rk(kernel, nodes[i], nodes[j])
            mat[j,i] = mat[i,j]
        end
    end
    m_1_p1 = m_1 + 1
    @inbounds for j = m_1_p1:m
        j1 = j - m_1
        for i = 1:m_1
            mat[i,j] = _∂rk_∂e(kernel, nodes[i], d_nodes[j1], es[j1])
            mat[j,i] = mat[i,j]
        end
    end
    ε2 = kernel.ε^2
    @inbounds for j = m_1_p1:m
        j1 = j - m_1
        for i = j:m
            if i == j
                mat[j,j] = ε2
                continue
            end
            i1 = i - m_1
            s  = zero(eltype(mat))
            for r = 1:n
                for k = 1:n
                    s += _∂²rk_∂η_r_∂ξ_k(kernel, d_nodes[j1], d_nodes[i1], r, k) * es[j1][k] * es[i1][r]
                end
            end
            mat[j,i] = s
            mat[i,j] = s
        end
    end
    return mat
end
function _gram(
        nodes::AbstractVectorOfSVectors{n},
        d_nodes::AbstractVectorOfSVectors{n},
        es::AbstractVectorOfSVectors{n},
        kernel::ReproducingKernel_1,
    ) where {n}
    T = promote_type(typeof(kernel.ε), eltype(eltype(nodes)), eltype(eltype(d_nodes)), eltype(eltype(es)))
    m_1 = length(nodes)
    m_2 = length(d_nodes)
    m = m_1 + m_2
    _gram!(zeros(T, m, m), nodes, d_nodes, es, kernel)
end
function _gram(nodes::AbstractMatrix, d_nodes::AbstractMatrix, es::AbstractMatrix, kernel::ReproducingKernel_1)
    _gram(to_vectors(nodes), to_vectors(d_nodes), to_vectors(es), kernel)
end
