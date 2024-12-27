#' Compute logFC as the difference of the median between two groups
#'
#' This function is intended to be part of step_logFC, as an extra recipe for
#' ML using tidymodels. It needs a training matrix with CPMs and a grouping
#' variable as columns, and samples as rows. It computes the median CPMs per
#' gene by group in `groups`, and then substracts the median CPM of the second
#' group to the median CPM of the first group.
#'
#' @param train_matrix dataframe or tibble, training matrix with CPMs and a
#' grouping variable as columns, and samples as rows.
#' @param groups two element vector with the two groups we want to compare
#' @param log boolean, should `log2()` be performed or not
#'
#' @return Description of output
#'
#' @examples
#' \dontrun{
#' compute_logFC(train_matrix, groups=c("PE", "NT"))
#' }
#'
#' @export
compute_logFC <- function(train_matrix, groups, log=FALSE) {
  checkmate::test_vector(groups, len = 2)
  checkmate::test_logical(log)
  
  medianCPM_matrix <- train_matrix %>%
    dplyr::group_by(group) %>%
    dplyr::summarise_all("median")
  
  medianCPM_matrix<- left_join(data.frame(group=groups),
                               medianCPM_matrix,
                               by="group")
  if (length(c(medianCPM_matrix)) > 2) {
    medianCPM_matrix <- as.data.frame(do.call(rbind, c(medianCPM_matrix[,-1])))
  } else {
    medianCPM_matrix <- as.data.frame(t(matrix(medianCPM_matrix[,-1])))
  }
  
  colnames(medianCPM_matrix) <- c("first", "second")
  
  #logFC_matrix <- medianCPM_matrix %>% mutate(FC = (abs(first) - abs(second))/abs(first))
  logFC_matrix <- medianCPM_matrix %>% mutate(FC = (abs(first) / abs(second)))
  if (log == TRUE) {
    logFC_matrix <- logFC_matrix %>% mutate(logFC = log2(FC))
  }
  
  logFC_matrix
}





