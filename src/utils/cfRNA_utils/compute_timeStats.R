#' Error skip t.test
#'
#' t.test implementation where errors due to constant data return NA instead of
#' an error
#'
#' @param Tdata A matrix. Rows should be samples, columns should be features. Last
#' column shoud be a grouping variable named "group" with two levels.
#'
#' @return pvalue
#'
#'
#' @examples
#' \dontrun{
#' my.t.test.p.value()
#' }
#'
#' @export
#' 
my.t.test.p.value <- function(...) {
  obj<-try(t.test(...), silent=TRUE)
  if (is(obj, "try-error")) return(NA) else return(obj$p.value)
}

#' Error skip wilcoxon
#'
#' wilcoxon implementation where errors due to constant data return NA instead of
#' an error
#'
#' @param Tdata A matrix. Rows should be samples, columns should be features. Last
#' column shoud be a grouping variable named "group" with two levels.
#'
#' @return pvalue
#'
#'
#' @examples
#' \dontrun{
#' my.t.test.p.value()
#' }
#'
#' @export
#' 
my.wilcox.test.p.value <- function(...) {
  obj<-try(wilcox.test(...), silent=TRUE)
  if (is(obj, "try-error")) return(NA) else return(obj$p.value)
}

#' t.test pval
#'
#'
#'
#' @param Tdata A matrix. Rows should be samples, columns should be features. Last
#' column should be a grouping variable named "group" with two levels.
#'
#' @return A named list of p-values
#'
#'
#' @examples
#' \dontrun{
#' compute_ttest_timePval(Tdata)
#' }
#'
#' @export
#' 
compute_ttest_timePval <- function(T_data)
{
  
  # Filter data ----------------------------------------------------------------
  
  
  nested <- T_data %>%
    summarise(across(where(is.numeric),  ~ list(
      my.t.test.p.value(.~group)
    ))) %>% as.vector() %>% unlist()
  
  return(nested)
}

#' Wilcoxon pval compute
#'
#'
#'
#' @param Tdata A matrix. Rows should be samples, columns should be features. Last
#' column should be a grouping variable named "group" with two levels.
#'
#' @return A named list of p-values
#'
#'
#' @examples
#' \dontrun{
#' compute_wilcox_timePval(Tdata)
#' }
#'
#' @export
#' 
compute_wilcox_timePval <- function(T_data)
{
  
  # Filter data ----------------------------------------------------------------
  
  
  nested <- T_data %>%
    summarise(across(where(is.numeric),  ~ list(
      my.wilcox.test.p.value(.~group, exact=F)
    ))) %>% as.vector() %>% unlist()
  
  return(nested)
}
