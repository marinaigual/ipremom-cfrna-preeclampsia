#' Combines several one-sample counts matrices into a single count matrix
#'
#' This function takes a folder `folder` given by the user and selects all the
#' counts matrices files, selected by a given `filePattern`. All the files are
#' combined into a single data.frame object, containing a GeneID column and
#' another column per file/sample. The sample names for the columns are
#' extracted from the file names using another given pattern, `samplePattern`.
#'
#' @param folder The folder path containing the counts matrices
#' @param filePattern A pattern to recognize the counts matrices
#' @param samplePattern A pattern to extract the sample names
#'
#' @return A matrix or data.frame object.
#'
#' @examples
#' \dontrun{
#' build_counts_matrix("./featureCounts_output/")
#' }
#'
#' @export
build_counts_matrix <- function(folder,
                                filePattern=".*counts.txt",
                                samplePattern = ".*-T[0-9]*")
{
  files <- list.files(path = folder, pattern = filePattern)
  samples <- str_extract(files, pattern = samplePattern)
  
  countsMatrix <- read_delim(file.path(folder,files[1]))
  colnames(countsMatrix) <- c("GeneID", files[1])
  
  for (file in files[2:length(files)]) {
    tmp <- read_delim(file.path(folder,file))
    colnames(tmp) <- c("GeneID", file)
    countsMatrix <- full_join(countsMatrix, tmp, by="GeneID")
  }
  colnames(countsMatrix) <- c("GeneID", samples)
  
  countsMatrix
}
