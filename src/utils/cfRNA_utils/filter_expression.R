#' Filters out counts CPMs
#'
#' This function removes poorly detected genes from a counts matrix
#' `countMatrix`. CPMs are computed, and genes with an expression level over
#' `CPMcutoff` across `minSamples` are kept for further analyses.
#'
#' @param countMatrix A dataframe or matrix object containing a counts matrix. 
#' Samples should be columns and genes should be rows.
#' @param CPMcutoff An integer value indicating the expression cutoff to be
#' used.
#' @param minSamples Proportion of samples to use as threshold
#'
#' @return Boolean vector indicating the genes to keep from the matrix
#'
#' @examples
#' \dontrun{
#' filter_median(countMatrix)
#' }
#'
#' @export
filter_expression <- function(countMatrix,
                          CPMcutoff=0.5,
                          minSamples=0.75)
{
  keep <- rowSums(edgeR::cpm(countMatrix) > CPMcutoff) >= (floor(ncol(countMatrix)*minSamples))
  
  keep
}