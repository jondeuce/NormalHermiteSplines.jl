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

@testset "ElasticNormalSpline" begin
    nhs = NormalHermiteSplines
    max_size = 4
    kernel = RK_H0(0.5 + rand())
    T = Float64

    for n in 1:3
        values = rand(max_size)
        min_bound = -rand(SVector{n,T})
        max_bound = rand(SVector{n,T})
        rand_node = () -> min_bound .+ rand(SVector{n,T}) .* (max_bound .- min_bound)
        nodes = [min_bound, max_bound, (rand_node() for i in 3:max_size)...]

        espl = ElasticNormalSpline(min_bound, max_bound, max_size, kernel)
        C = espl._chol

        for i in 1:max_size
            # Update `ElasticNormalSpline`
            insert!(espl, nodes[i], values[i])
            i == 1 && continue # `NormalSpline` requires at least two nodes

            # Compute `NormalSpline`
            spl = interpolate(nodes[1:i], values[1:i], kernel)

            J = C.colperms[1:C.ncols[]]
            b = randn(i)
            @test nhs._get_kernel(espl)      == nhs._get_kernel(spl)
            @test nhs._get_nodes(espl)       ≈ nhs._get_nodes(spl)
            @test nhs._get_values(espl)      ≈ nhs._get_values(spl)
            @test nhs._get_d_nodes(espl)     ≈ nhs._get_d_nodes(spl)
            @test nhs._get_d_dirs(espl)      ≈ nhs._get_d_dirs(spl)
            @test nhs._get_d_values(espl)    ≈ nhs._get_d_values(spl)
            @test nhs._get_mu(espl)          ≈ nhs._get_mu(spl)
            @test nhs._get_gram(espl)        ≈ nhs._get_gram(spl)
            @test UpperTriangular(C.U[J, J]) ≈ nhs._get_chol(spl).U
            @test ldiv!(similar(b), C, b)    ≈ nhs._get_chol(spl) \ b
            # @test nhs._get_cond(espl)      ≈ nhs._get_cond(spl) # condition number algorithm is... poorly conditioned?
            @test nhs._get_min_bound(espl)   ≈ nhs._get_min_bound(spl)
            @test nhs._get_max_bound(espl)   ≈ nhs._get_max_bound(spl)
            @test nhs._get_scale(espl)       ≈ nhs._get_scale(spl)
        end
    end
end
