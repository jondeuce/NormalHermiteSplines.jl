function _gram!(mat::AbstractMatrix, nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    n_1 = length(nodes)
    @inbounds for l = 1:size(mat, 2)
        for i = 1:l
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
    @inbounds for j = 1:n_1
        for i = 1:j
            mat[i,j] = _rk(kernel, nodes[i], nodes[j])
            mat[j,i] = mat[i,j]
        end
    end
    @inbounds for j = n_1 + 1:n_1 + n_2
        j1 = j - n_1
        for i = 1:n_1
            mat[i,j] = _∂rk_∂e(kernel, nodes[i], d_nodes[j1], es[j1])
            mat[j,i] = mat[i,j]
        end
    end
    ε2 = kernel.ε^2
    @inbounds for j = n_1 + 1:n_1 + n_2
        j1 = j - n_1
        for i = j:n_1 + n_2
            if i == j
                mat[j,j] = ε2
                continue
            end
            i1 = i - n_1
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
