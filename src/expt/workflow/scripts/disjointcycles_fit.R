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
# `est_ord` is list-of-layers; each layer is a list of integer vectors,
# where each vector is the node set of one SCC (singleton or cycle).
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
