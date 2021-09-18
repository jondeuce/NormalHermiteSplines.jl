@testset "ElasticCholesky" begin
    T = Float64
    max_size = 4
    A = rand(MersenneTwister(0), max_size, max_size)
    A = A'A

    # Test incrementally adding columns of `A` to `ElasticCholesky`
    for colperms in [[1,2,3,4], [4,3,2,1], [1,3,2,4], [4,1,3,2]]
        C = ElasticCholesky{T}(max_size)
        for j in 1:max_size
            # Add column `j` of `A` to the `jperm`th column of `C.A`
            jperm = colperms[j]
            v = A[1:j,j]
            cholesky!(C, jperm, v, Val(true))

            # Check the fields of `C`
            J = colperms[1:C.ncols[]]
            @test Hermitian(C.A[J, J], :U) ≈ A[1:j, 1:j]
            @test UpperTriangular(C.U[J, J]) ≈ cholesky(A[1:j, 1:j]).U
            @test C.colperms[1:C.ncols[]] == J
            @test C.ncols[] == j

            # Check `ldiv!` gives same result as `LinearAlgebra.cholesky`
            b = rand(j)
            @test ldiv!(similar(b), C, b) ≈ cholesky(A[1:j, 1:j]) \ b
        end
    end

    # Test incremental factorization of wrapped array
    C = ElasticCholesky(copy(A))
    cholesky!(C)

    @test Hermitian(C.A, :U) ≈ A
    @test UpperTriangular(C.U) ≈ cholesky(A).U
    @test C.colperms == 1:max_size
    @test C.ncols[] == max_size

    b = rand(max_size)
    @test ldiv!(similar(b), C, b) ≈ cholesky(A) \ b
end

function random_nodes(n, T, max_size)
    min_bound   = -rand(SVector{n,T})
    max_bound   = rand(SVector{n,T})
    rand_node() = min_bound .+ rand(SVector{n,T}) .* (max_bound .- min_bound)
    rand_dir()  = (x = rand_node(); x/norm(x))

    nodes     = [min_bound, max_bound, (rand_node() for i in 3:max_size)...]
    values    = rand(max_size)
    d_nodes   = [rand_node() for _ in 1:n*max_size]
    d_dirs    = [rand_dir() for _ in 1:n*max_size]
    d_values  = rand(n*max_size)
    return (; min_bound, max_bound, nodes, values, d_nodes, d_dirs, d_values)
end

@testset "Elastic Gram Matrix" begin
    nhs = NormalHermiteSplines
    max_size = 4
    T = Float64

    @testset "RK_H0" begin
        kernel = RK_H0(0.5 + rand())
        for n in 1:3
            # Test incremental building of Gram matrix, inserting nodes in random order
            @unpack nodes = random_nodes(n, T, max_size)
            n₁ = length(nodes)
            A  = nhs._gram(nodes, kernel)
            A′ = zeros(T, n₁, n₁)
            J  = Int[]
            for j in randperm(n₁)
                push!(J, j)
                nhs._gram!(view(A′, J, J), nodes[J[end]], nodes[J[1:end-1]], kernel)
                @test Hermitian(A′[J,J], :U) ≈ A[J,J] # `A′[J,J]` is upper triangular, `A` is Hermitian
            end
        end
    end

    @testset "RK_H1" begin
        kernel = RK_H1(0.5 + rand())
        for n in 1:3
            # Test incremental building of Gram matrix, inserting (derivative-)nodes in random order
            @unpack nodes, d_nodes, d_dirs = random_nodes(n, T, max_size)
            n₁, n₂ = length(nodes), length(d_nodes)
            A  = nhs._gram(nodes, d_nodes, d_dirs, kernel)
            A′ = zeros(T, n₁+n₂, n₁+n₂)
            J, J₁, J₂ = Int[], Int[], Int[]
            for j in randperm(n₁+n₂)
                if j <= n₁
                    push!(J₁, j)
                    insert!(J, length(J₁), j)
                    nhs._gram!(view(A′, J, J), nodes[J₁[end]], nodes[J₁[1:end-1]], d_nodes[J₂], d_dirs[J₂], kernel)
                else
                    push!(J₂, j-n₁)
                    push!(J, j)
                    nhs._gram!(view(A′, J, J), d_nodes[J₂[end]], d_dirs[J₂[end]], nodes[J₁], d_nodes[J₂[1:end-1]], d_dirs[J₂[1:end-1]], kernel)
                end
                @test Hermitian(A′[J,J], :U) ≈ A[J,J] # `A′[J,J]` is upper triangular, `A` is Hermitian
            end
        end
    end
end

@testset "ElasticNormalSpline" begin
    nhs = NormalHermiteSplines
    max_size = 3
    T = Float64

    for n in 1:3
        @unpack min_bound, max_bound, nodes, values, d_nodes, d_dirs, d_values = random_nodes(n, T, max_size)
        rk_H0 = RK_H0(0.5 + rand())
        rk_H1 = RK_H1(0.5 + rand())
        global espl_H0 = ElasticNormalSpline(min_bound, max_bound, max_size, rk_H0)
        global espl_H1_0 = ElasticNormalSpline(min_bound, max_bound, max_size, rk_H1)
        global espl_H1_1 = ElasticNormalSpline(min_bound, max_bound, max_size, rk_H1)

        for i in 1:max_size
            # Update `ElasticNormalSpline`
            insert!(espl_H0, nodes[i], values[i])
            insert!(espl_H1_0, nodes[i], values[i])
            insert!(espl_H1_1, nodes[i], values[i])
            insert!(espl_H1_1, d_nodes[i], d_dirs[i], d_values[i])
            i == 1 && continue # `NormalSpline` requires at least two nodes
            # i==1 && (println("\n--------------------------------------------------\n"); continue)

            # Compute `NormalSpline`
            global spl_H0  = interpolate(nodes[1:i], values[1:i], rk_H0)
            global spl_H1_0 = interpolate(nodes[1:i], values[1:i], rk_H1)
            global spl_H1_1 = interpolate(nodes[1:i], values[1:i], d_nodes[1:i], d_dirs[1:i], d_values[1:i], rk_H1)
            # # println("NormalSpline: (RK_H1, n₁=$max_size, n₂=$max_size)"); display(UpperTriangular(parent(nhs._get_gram(spl_H1_1)))); println("\n")
            # println("\n--------------------------------------------------\n")

            # println("spl_H1_0"); display(nhs._get_mu(spl_H1_0)'); display(nhs._get_rhs(spl_H1_0)'); println("")
            # println("espl_H1_0"); display(nhs._get_mu(espl_H1_0)'); display(nhs._get_rhs(espl_H1_0)'); println("")

            # println("spl_H1_1"); display(nhs._get_mu(spl_H1_1)'); display(nhs._get_rhs(spl_H1_1)'); println("")
            # println("espl_H1_1"); display(nhs._get_mu(espl_H1_1)'); display(nhs._get_rhs(espl_H1_1)'); println("")

            for (espl, spl) in [
                    (espl_H0, spl_H0)
                    (espl_H1_0, spl_H1_0)
                    (espl_H1_1, spl_H1_1)
                ]
                C = espl._chol
                J = C.colperms[1:C.ncols[]]
                # J = nhs._get_block_indices(espl)
                b = randn(i)
                @test nhs._get_kernel(espl)      == nhs._get_kernel(spl)
                @test nhs._get_nodes(espl)       ≈ nhs._get_nodes(spl)
                @test nhs._get_values(espl)      ≈ nhs._get_values(spl)
                @test nhs._get_d_nodes(espl)     ≈ nhs._get_d_nodes(spl)
                @test nhs._get_d_dirs(espl)      ≈ nhs._get_d_dirs(spl)
                @test nhs._get_d_values(espl)    ≈ nhs._get_d_values(spl)
                # @test nhs._get_mu(espl)          ≈ nhs._get_mu(spl)
                @test nhs._get_gram(espl)        ≈ nhs._get_gram(spl)
                try
                    @assert UpperTriangular(C.U[J, J]) ≈ nhs._get_chol(spl).U
                    # @test cholesky(copy(nhs._get_gram(espl))).U ≈ nhs._get_chol(spl).U
                catch
                    @show J'
                    println("$(typeof(espl))"); display(C.U[J, J]); println("")
                    println("$(typeof(spl))"); display(nhs._get_chol(spl).U); println("")
                end
                # @test ldiv!(similar(b), C, b)    ≈ nhs._get_chol(spl) \ b
                # @test nhs._get_cond(espl)      ≈ nhs._get_cond(spl)
                @test nhs._get_min_bound(espl)   ≈ nhs._get_min_bound(spl)
                @test nhs._get_max_bound(espl)   ≈ nhs._get_max_bound(spl)
                @test nhs._get_scale(espl)       ≈ nhs._get_scale(spl)
            end
        end
    end
end
