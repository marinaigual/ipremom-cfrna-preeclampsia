#' Applies VST normalization to a dataset
#'
#' This function takes a counts matrix `countMatrix` and computes the VST
#' normalization implemented by DESeq2
#'
#' @param countData A dataframe or matrix object containing a counts matrix. 
#' Samples should be columns and genes should be rows.
#' @param metaData A DataFrame or data.frame with at least a single column.
#' Rows of colData correspond to columns of countData
#' @param formula A formula indicating the design of the experiment.
#' The formula expresses how the counts for each gene depend on the variables
#' in colData. Many R formula are valid, including designs with multiple
#' variables, e.g., ~ group + condition, and designs with interactions, e.g.,
#' ~ genotype + treatment + genotype:treatment. 
#' @param tidy for matrix input: whether the first column of countData is the
#' rownames for the count matrix
#'
#' @return Large DESeqTransform.
#'
#' @examples
#' \dontrun{
#' norm_vst(countData, colData)
#' }
#'
#' @export
norm_vst <- function(countData,
                     colData,
                     formula= ~ 1)
{
  dds <- DESeq2::DESeqDataSetFromMatrix(countData = countData,
                                        colData = colData,
                                        formula)
  vsd <- DESeq2::vst(dds, blind = FALSE)
  
  vsd
}