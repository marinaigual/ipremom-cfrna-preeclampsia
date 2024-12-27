#' Filters out counts based on an expression cutoff value
#'
#' This function takes a counts matrix `countMatrix` and computes the median of
#' Log1 CPMs. These vales are used with an expression cutoff `exprCutoff` value
#' to filter out genes with low expression.
#'
#' @param countMatrix A dataframe or matrix object containing a counts matrix. 
#' Samples should be columns and genes should be rows.
#' @param exprCutoff An integer value indicating the expression cutoff to be
#' used.
#'
#' @return A matrix or dataframe object.
#'
#' @examples
#' \dontrun{
#' filter_median(countMatrix)
#' }
#'
#' @export
filter_median <- function(countMatrix,
                          exprCutoff=-1)
{
  cpmLog <- edgeR::cpm(countMatrix, log = TRUE)
  medianLog2Cpm <- apply(cpmLog, 1, median)
  
  fCounts <- countMatrix[medianLog2Cpm > exprCutoff, ]
  
  fCounts
}