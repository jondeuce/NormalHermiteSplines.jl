using LibGit2: LibGit2

const pkg_folder = normpath(@__DIR__, "../..")
const src_folder = joinpath(pkg_folder, "src")
const test_folder = joinpath(pkg_folder, "test")

function mergefiles(mainfile, depfiles)
    repo = LibGit2.GitRepo(pkg_folder)
    hash = LibGit2.GitHash(LibGit2.peel(LibGit2.GitCommit, LibGit2.head(repo))) |> string

    main = """
    #=
    AUTO-GENERATED FILE - DO NOT EDIT

    This file is derived from the following fork of the NormalHermiteSplines.jl package:

        https://github.com/jondeuce/NormalHermiteSplines.jl#$(hash)

    As it is not possible to depend on a package fork, the above module is included here verbatim.

    The `LICENSE.md` file contents from the original repository follows:

    ################################################################################

    MIT License

    Copyright (c) 2021 Igor Kohanovsky

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    =#

    $(readchomp(mainfile))
    """
    for (i, depfile) in enumerate(depfiles)
        maybenewline = i == length(depfiles) ? "" : "\n"
        main = replace(
            main,
            "include(\"$(basename(depfile))\")" => chomp(
                """
                ####
                #### $(basename(depfile))
                ####

                $(readchomp(depfile))
                """,
            ) * maybenewline,
        )
    end
    return main
end

for (dir, (mainfile, depfiles)) in [
    src_folder => "NormalHermiteSplines.jl" => [
        "ReproducingKernels.jl",
        "GramMatrix.jl",
        "Splines.jl",
        "Utils.jl",
        "Interpolate.jl",
    ]
    test_folder => "runtests.jl" => [
        "1D.jl",
        "2D.jl",
        "3D.jl",
        "elastic.jl",
        "derivatives.jl",
    ]
]
    outdir = isempty(ARGS) ? pwd() : ARGS[1]
    outfile = joinpath(outdir, mainfile)
    open(outfile; write = true) do io
        # Make sure we write 64bit integer in little-endian byte order
        return write(io, mergefiles(joinpath(dir, mainfile), joinpath.(dir, depfiles)))
    end
end
