function _gram!(mat::AbstractMatrix, nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    n_1 = length(nodes)
    @inbounds for l in 1:size(mat, 2)
        for i in 1:l
            mat[i,l] = _rk(kernel, nodes[i], nodes[l])
            mat[l,i] = mat[i,l]
        end
    end
    return mat
end

function _gram!(
        mat::AbstractMatrix,
        nodes::AbstractVecOfSVecs{n},
        d_nodes::AbstractVecOfSVecs{n},
        es::AbstractVecOfSVecs{n},
        kernel::ReproducingKernel_1,
    ) where {n}
    n_1 = length(nodes)
    n_2 = length(d_nodes)
    @inbounds for j in 1:n_1
        for i in 1:j
            mat[i,j] = _rk(kernel, nodes[i], nodes[j])
            mat[j,i] = mat[i,j]
        end
    end
    @inbounds for j in n_1 + 1:n_1 + n_2
        j1 = j - n_1
        for i in 1:n_1
            mat[i,j] = _∂rk_∂e(kernel, nodes[i], d_nodes[j1], es[j1])
            mat[j,i] = mat[i,j]
        end
    end
    ε² = kernel.ε^2
    @inbounds for j in n_1 + 1:n_1 + n_2
        j1 = j - n_1
        for i in j:n_1 + n_2
            if i == j
                mat[j,j] = ε²
            else
                i1 = i - n_1
                ∂²rk = es[i1] ⋅ (_∂²rk_∂η∂ξ(kernel, d_nodes[j1], d_nodes[i1]) * es[j1])
                mat[j,i] = ∂²rk
                mat[i,j] = ∂²rk
            end
        end
    end
    return mat
end
