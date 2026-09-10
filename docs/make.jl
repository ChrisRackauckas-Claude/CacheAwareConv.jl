using Documenter, CacheAwareConv

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

include("pages.jl")

makedocs(
    sitename = "CacheAwareConv.jl",
    authors = "Oscar Smith",
    modules = [CacheAwareConv],
    repo = Documenter.Remotes.GitHub("SciML", "CacheAwareConv.jl"),
    clean = true, doctest = true, checkdocs = :exports, linkcheck = true,
    warnonly = [:missing_docs, :linkcheck],
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/CacheAwareConv/stable/"
    ),
    pages = pages
)

deploydocs(repo = "github.com/SciML/CacheAwareConv.jl.git"; push_preview = true)
