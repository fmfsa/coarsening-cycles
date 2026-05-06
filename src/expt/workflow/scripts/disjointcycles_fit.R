#!/usr/bin/env Rscript
# Wrapper around the `disjointCycles` R package (Drton, Garrote-Lopez,
# Nikov, Robeva, Wang). Called from fit.py via subprocess.
#
# Args:
#   1. obs_csv     — path to (n, d) observation matrix CSV (no header, no row names)
#   2. alpha       — significance level (numeric, e.g. 0.01)
#   3. adj_out     — path to write (d, d) i->j adjacency CSV (no header)
#   4. ord_out     — path to write JSON ordering: list of SCC clusters (0-indexed)

suppressPackageStartupMessages({
  library(disjointCycles)
  library(jsonlite)
})

# -----------------------------------------------------------------------------
# Compatibility patch.
#
# `disjointCycles:::djc_recurNew` builds a p×p `det3` matrix as `matrix(NA, p, p)`,
# fills only the upper triangle with adjusted p-values, and then calls
# `igraph::graph_from_adjacency_matrix(det3 >= threshold3, mode = "upper")`.
# Newer `igraph` (≥ 2.0) refuses to construct a graph from a matrix that
# contains NA, even with `mode = "upper"`. Older `igraph` tolerated it,
# which is what the package was tested against. The same pattern repeats
# for `temp_Adj` later in the same function.
#
# We patch both call sites in-process to coerce NA → FALSE before passing
# to igraph. No fork of the upstream package required.
# -----------------------------------------------------------------------------
.dc_src <- paste(deparse(disjointCycles:::djc_recurNew), collapse = "\n")
.dc_src <- gsub(
  "igraph::graph_from_adjacency_matrix(det3 >= threshold3,",
  "igraph::graph_from_adjacency_matrix({ .m <- det3 >= threshold3; .m[is.na(.m)] <- FALSE; .m },",
  .dc_src, fixed = TRUE
)
.dc_src <- gsub(
  "igraph::graph_from_adjacency_matrix(temp_Adj,",
  "igraph::graph_from_adjacency_matrix({ .m <- temp_Adj; .m[is.na(.m)] <- FALSE; .m },",
  .dc_src, fixed = TRUE
)
.dc_patched <- eval(parse(text = .dc_src))
environment(.dc_patched) <- asNamespace("disjointCycles")
assignInNamespace("djc_recurNew", .dc_patched, "disjointCycles")
rm(.dc_src, .dc_patched)

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
obs_csv <- args[[1]]
alpha   <- as.numeric(args[[2]])
adj_out <- args[[3]]
ord_out <- args[[4]]

Y <- as.matrix(read.csv(obs_csv, header = FALSE))

est_ord <- djcGetOrderNew(
  Y,
  alpha2 = alpha, alpha3 = alpha, alphaR = alpha,
  pvalAdjMethod = "holm",
  methodPR = "chisq",
  rescaleData = TRUE,
  verbose = FALSE
)

est_edges <- djcGetEdges(
  est_ord, Y,
  alpha = alpha,
  pvalAdjMethod = "holm"
)

adj <- est_edges$adjMat
diag(adj) <- 0
write.table(adj, file = adj_out, sep = ",",
            row.names = FALSE, col.names = FALSE)

# Flatten the layered ordering into a flat list of clusters (SCCs).
clusters <- list()
for (layer in est_ord) {
  if (is.list(layer)) {
    for (grp in layer) {
      clusters[[length(clusters) + 1L]] <- as.integer(grp) - 1L
    }
  } else {
    clusters[[length(clusters) + 1L]] <- as.integer(layer) - 1L
  }
}

writeLines(toJSON(clusters, auto_unbox = FALSE), ord_out)
