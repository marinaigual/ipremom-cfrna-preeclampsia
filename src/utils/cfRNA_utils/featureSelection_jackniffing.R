#' Feature selection - Jackniffing optimization
#'
#'Perform differential expression analysis with limma, focus on sampling periods
#'and return list of candidate genes for BAS or modeling.
#'
#' @param expressionMatrix A matrix. Columns and rows should be named.
#' @param colData Sample metadata dataframe. It must contain at least a grouping
#' variable. Rownames must be equal to `expressionMatrix` colnames.
#' @param folder Folder where results should be stored
#' @param group_var Name of the grouping variable in `colData`
#' @param time_var Name of the time variable in `colData`
#' @param patient_var Name of the variable storing patient codes in `colData`
#' @param spline_config Provide the degrees of freedom or knots for a spline
#' constructed with `splines::ns()`. If a single integer is provided, it will be
#' passed as degrees of freedom. If two or more are provided, they will be
#' passed as knots. 
#'
#' @return A named list of Benjamini & Hockberg corrected p-values
#'
#'
#' @examples
#' \dontrun{
#' featureSelection_jackniffing(countMatrix, colData, group_var, time_var, patient_var)
#' }
#'
#' @export
featureSelection_jackniffing <- function(samples, 
                                         expressionMatrix,
                                         colData,
                                         folder,
                                         group_var,
                                         time_var,
                                         patient_var,
                                         spline_config = c(112, 154, 186)
                                         )
  {
  # Initial checkings ----------------------------------------------------------
  collection <- checkmate::makeAssertCollection()
  checkmate::assert(
    checkmate::checkCharacter(samples, na.ok = F),
    checkmate::check_subset(colnames(expressionMatrix), rownames(colData)),
    checkmate::checkString(folder, na.ok = F),
    checkmate::check_integer(spline_config),
    checkmate::checkString(group_var, na.ok = F),
    checkmate::checkString(time_var, na.ok = F),
    checkmate::checkString(patient_var, na.ok = F),
    add = collection,
    combine = "and")
  collection$getMessages()
  
  # Prepare data ---------------------------------------------------------------
  colData <- colData[samples,]
  expressionMatrix <- expressionMatrix[,samples]
  # Differential expression analysis
  message("Starting differential expression analysis")
  de.results <- diffexp_limma(expressionMatrix,
                              colData,
                              group_var,
                              time_var,
                              patient_var,
                              spline_config)
  message("Saving differential expression results")
  de_file_name <- paste0(folder, "/jackniffing_DEresults.tsv")
  de_tsv <- file.exists(metrics_file_name)
  de_table <- as.data.frame(de.results)
  
  if (de_tsv) {
    de_file = read_delim(de_file_name,
                              delim = "\t") %>%
      bind_cols(de_table, .name_repair = "unique") %>%
      write_delim(file = de_file_name,
                  delim = "\t")
    } else {
    if (!dir.exists(folder)){
      dir.create(folder)
    }
    write_delim(de_table,
                file = de_file_name,
                delim = "\t")
    }
  
  significant_DE <- de_table %>%
    tibble::rownames_to_column(var="gene") %>%
    filter(x < 0.05) %>%
    pull(gene)
  
  
  # Focus on first trimester ---------------------------------------------------
  
  T_samples <- colData %>%
    dplyr::filter(toma == "T1")
  T_group <- T_samples[, group_var] %>% pull()
  
  T_data <- expressionMatrix[,rownames(T_samples)]
  T_data <- cbind(data.frame(t(T_data)),
                  group=T_group)
  
  FT_pvals <- compute_timePval(T_data)
  genes_pvals <- names(FT_pvals[FT_pvals < 0.05])
  FT_adjs <- p.adjust(FT_pvals, method = "BH")
  
  FT_Fc <- compute_logFC(T_data[,c(genes_pvals, "group")], 
                         groups = c("PE", "Control"),
                         log = TRUE)
  genes_Fc <- FT_Fc %>%
    tibble::rownames_to_column(var = "genes") %>%
    dplyr::filter(abs(logFC > 0.2)) %>%
    dplyr::pull(genes)
  
  }