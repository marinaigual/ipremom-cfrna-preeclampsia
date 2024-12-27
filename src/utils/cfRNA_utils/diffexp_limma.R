#' Differential expression analysis with limma - Jackniffing-spline optimization
#'
#'Perform differential expression analysis with limma, following the voom
#'approach with time modeling as splines. Data is TMM normalized. Spline
#'modelling is performed based on the `spline_config` parameter. The grouping
#'variable in `colData`, `group_var`, is assumed to have two levels, case and
#'control. The contrast will compare the first level with the second, e.g. case
#'- control. 
#'
#' @param expressionMatrix A matrix. Columns and rows should be named.
#' @param colData Sample metadata dataframe. It must contain at least a grouping
#' variable. Rownames must be equal to `expressionMatrix` colnames.
#' @param group_var Name of the grouping variable in `colData`
#' @param time_var Name of the time variable in `colData`
#' @param patient_var Name of the variable storing patient codes in `colData`
#' @param spline_config Provide the degrees of freedom or knots for a spline
#' constructed with `splines::ns()`. If a single integer is provided, it will be
#' passed as degrees of freedom. If two or more are provided, they will be
#' passed as knots. 
#'
#' @return Limma topTable output for timepoints analysis
#'
#'@import dplyr, splines, edgeR, limma, checkmate
#'
#' @examples
#' \dontrun{
#' diffexp_limma_spline(countMatrix, colData, group, time)
#' }
#'
#' @export
diffexp_limma_spline <- function(expressionMatrix,
                                 colData,
                                 group_var,
                                 time_var,
                                 patient_var,
                                 spline_config = c(112, 154, 186)
)
{
  # Initial checkings ----------------------------------------------------------
  collection <- checkmate::makeAssertCollection()
  checkmate::assert(
    checkmate::check_subset(colnames(expressionMatrix), rownames(colData)),
    checkmate::check_integer(spline_config),
    checkmate::checkString(group_var, na.ok = F),
    checkmate::checkString(time_var, na.ok = F),
    checkmate::checkString(patient_var, na.ok = F),
    add = collection,
    combine = "and")
  collection$getMessages()
  
  # Prepare data ---------------------------------------------------------------
  
  colData$group <- colData[,group_var] %>% pull()
  patient_info <- colData[,patient_var] %>% pull()
  dge_pe <- edgeR::DGEList(expressionMatrix)
  dge_tmm <- edgeR::calcNormFactors(dge_pe, method = "TMM")
  message("Using TMM normalized data")
  
  # Prepare design -------------------------------------------------------------
  
  if (length(spline_config) == 1) {
    time_spline = splines::ns(colData[,time_var] %>% pull(), df = spline_config)
    message("Modelling spline with ",
            attr(time_spline, which = "degree"),
            " degrees of freedom")
  } else {
    time_spline = splines::ns(colData[,time_var] %>% pull(), knots = spline_config)
    message("Modelling spline with knots " ,
            paste0(attr(time_spline, which = "knots")
                   , collapse = ", ")
    )
  }
  
  
  design <- model.matrix(~group*time_spline, data = colData)
  colnames(design) = make.names(colnames(design), unique = T)
  n_reps <- length(attr(time_spline, which = "knots")) + 1
  colnames(design) <- c(colnames(design)[1],
                        "secondLevel",
                        paste0(rep(c("time_spline",
                                     "secondLevel_time_spline"),
                                   each = n_reps),
                               rep(1:n_reps, 2)))
  
  # limma voom -----------------------------------------------------------------
  message("voom with quality weights")
  all_voom_1 = limma::voomWithQualityWeights(
    dge_tmm,
    design,
    normalize.method = "none",
    method = "genebygene",
    maxiter = 100,
    tol = 1e-6,
    trace = F,
    plot = F
  )
  
  message("Computing correlation")
  
  corfit = limma::duplicateCorrelation(all_voom_1, design, block = patient_info)
  
  message("voom with quality weights")
  
  all_voom_2 = limma::voomWithQualityWeights(
    dge_tmm,
    design,
    normalize.method = "none",
    method = "genebygene",
    maxiter = 100,
    tol = 1e-6,
    block = patient_info,
    correlation = corfit$consensus,
    trace = F,
    plot = F
  )
  
  fit_noBayes = lmFit(all_voom_2,
                      design,
                      block = patient_info,
                      correlation = corfit$consensus
  )
  
  fit = eBayes(fit_noBayes, trend = TRUE, robust = TRUE)
  
  # AnyTimepoint ---------------------------------------------------------------
  
  if (length(spline_config) > 2) {
    pe_v_control_cont = makeContrasts(
      secondLevel,
      time_spline1 - secondLevel_time_spline1,
      time_spline2 - secondLevel_time_spline2,
      time_spline3 - secondLevel_time_spline3,
      time_spline4 - secondLevel_time_spline4,
      levels = design
    )
  } else {
    pe_v_control_cont = makeContrasts(
      secondLevel,
      time_spline1 - secondLevel_time_spline1,
      time_spline2 - secondLevel_time_spline2,
      time_spline3 - secondLevel_time_spline3,
      levels = design
    )
  }
  
  
  fit_pe_v_control = contrasts.fit(fit_noBayes, pe_v_control_cont)
  fit_pe_v_control = eBayes(fit_pe_v_control)
  
  dTime_pe_v_control_anyTimepoint = topTable(fit_pe_v_control, sort.by="B", resort.by = "logFC", number = Inf, confint = TRUE) %>%
    rownames_to_column(var = "gene")
  
  return(dTime_pe_v_control_anyTimepoint)
  
}


#' Differential expression analysis with limma - Jackniffing traditional optimization
#'
#'Perform differential expression analysis with limma, following the voom
#'approach. Data is TMM normalized. The grouping
#'variable in `colData`, `group_var`, is assumed to have two levels, case and
#'control. The contrast will compare the first level with the second, e.g. case
#'- control. 
#'
#' @param expressionMatrix A matrix. Columns and rows should be named.
#' @param colData Sample metadata dataframe. It must contain at least a grouping
#' variable. Rownames must be equal to `expressionMatrix` colnames.
#' @param group_var Name of the grouping variable in `colData`
#' @param time_var Name of the time variable in `colData`
#' @param patient_var Name of the variable storing patient codes in `colData`
#'
#' @return Limma ebayes
#'
#'@import dplyr, edgeR, limma, checkmate
#'
#' @examples
#' \dontrun{
#' diffexp_limma_voomQW(countMatrix, colData, group, time)
#' }
#'
#' @export
diffexp_limma_voomQW <- function(expressionMatrix,
                                 colData,
                                 group_var,
                                 time_var,
                                 patient_var
)
{
  # Initial checkings ----------------------------------------------------------
  collection <- checkmate::makeAssertCollection()
  checkmate::assert(
    checkmate::check_subset(colnames(expressionMatrix), rownames(colData)),
    checkmate::checkString(group_var, na.ok = F),
    checkmate::checkString(patient_var, na.ok = F),
    add = collection,
    combine = "and")
  collection$getMessages()
  
  # Prepare data ---------------------------------------------------------------
  
  colData$group <- colData[,group_var] %>% pull()
  colData$time <- colData[,time_var] %>% pull()
  patient_info <- colData[,patient_var] %>% pull()
  dge_pe <- edgeR::DGEList(expressionMatrix)
  dge_tmm <- edgeR::calcNormFactors(dge_pe, method = "TMM")
  message("Using TMM normalized data")
  
  # Prepare design -------------------------------------------------------------
  
  colData$condTime <- factor(paste0(colData$group, colData$time),
                             levels = c("PreeclampsiaT1",
                                        "ControlT1",
                                        "PreeclampsiaT2",
                                        "ControlT2",
                                        "PreeclampsiaT3",
                                        "ControlT3"))
  design <- model.matrix(~0 + condTime, data = colData)
  colnames(design) = make.names(colnames(design), unique = T)
  
  # limma voom -----------------------------------------------------------------
  message("voom with quality weights")
  all_voom_1 = limma::voomWithQualityWeights(
    dge_tmm,
    design,
    normalize.method = "none",
    method = "genebygene",
    maxiter = 100,
    tol = 1e-6,
    trace = F,
    plot = F
  )
  
  message("Computing correlation")
  
  corfit = limma::duplicateCorrelation(all_voom_1, design, block = patient_info)
  
  message("voom with quality weights")
  
  all_voom_2 = limma::voomWithQualityWeights(
    dge_tmm,
    design,
    normalize.method = "none",
    method = "genebygene",
    maxiter = 100,
    tol = 1e-6,
    block = patient_info,
    correlation = corfit$consensus,
    trace = F,
    plot = F
  )
  
  fit_noBayes = lmFit(all_voom_2,
                      design,
                      block = patient_info,
                      correlation = corfit$consensus
  )
  
  fit = eBayes(fit_noBayes, trend = TRUE, robust = TRUE)
  
  
  # AnyTimepoint ---------------------------------------------------------------
  
  pe_v_control_cont = makeContrasts(
    condTimePreeclampsiaT1 - condTimeControlT1,
    condTimePreeclampsiaT2 - condTimeControlT2,
    condTimePreeclampsiaT3 - condTimeControlT3,
    levels = design
  )
  
  
  fit_pe_v_control = contrasts.fit(fit_noBayes, pe_v_control_cont)
  fit_pe_v_control = eBayes(fit_pe_v_control)
  
  
  return(fit_pe_v_control)
  
}


#' Differential expression analysis with limma - Jackniffing traditional optimization
#'
#'Perform differential expression analysis with limma, following the voom
#'approach. Data is TMM normalized. The grouping
#'variable in `colData`, `group_var`, is assumed to have two levels, case and
#'control. The contrast will compare the first level with the second, e.g. case
#'- control. 
#'
#' @param expressionMatrix A matrix. Columns and rows should be named.
#' @param colData Sample metadata dataframe. It must contain at least a grouping
#' variable. Rownames must be equal to `expressionMatrix` colnames.
#' @param group_var Name of the grouping variable in `colData`
#' @param time_var Name of the time variable in `colData`
#' @param patient_var Name of the variable storing patient codes in `colData`
#'
#' @return Limma ebayes
#'
#'@import dplyr, edgeR, limma, checkmate
#'
#' @examples
#' \dontrun{
#' diffexp_limma_voom(countMatrix, colData, group, time)
#' }
#'
#' @export
diffexp_limma_voom <- function(expressionMatrix,
                               colData,
                               group_var,
                               time_var,
                               patient_var
)
{
  # Initial checkings ----------------------------------------------------------
  collection <- checkmate::makeAssertCollection()
  checkmate::assert(
    checkmate::check_subset(colnames(expressionMatrix), rownames(colData)),
    checkmate::checkString(group_var, na.ok = F),
    checkmate::checkString(patient_var, na.ok = F),
    add = collection,
    combine = "and")
  collection$getMessages()
  
  # Prepare data ---------------------------------------------------------------
  
  colData$group <- colData[,group_var] %>% pull()
  colData$time <- colData[,time_var] %>% pull()
  patient_info <- colData[,patient_var] %>% pull()
  dge_pe <- edgeR::DGEList(expressionMatrix)
  dge_tmm <- edgeR::calcNormFactors(dge_pe, method = "TMM")
  message("Using TMM normalized data")
  
  # Prepare design -------------------------------------------------------------
  
  colData$condTime <- factor(paste0(colData$group, colData$time),
                             levels = c("PreeclampsiaT1",
                                        "ControlT1",
                                        "PreeclampsiaT2",
                                        "ControlT2",
                                        "PreeclampsiaT3",
                                        "ControlT3"))
  design <- model.matrix(~0 + condTime, data = colData)
  colnames(design) = make.names(colnames(design), unique = T)
  
  # limma voom -----------------------------------------------------------------
  # message("voom with quality weights")
  # all_voom_1 = limma::voomWithQualityWeights(
  #   dge_tmm,
  #   design,
  #   normalize.method = "none",
  #   method = "genebygene",
  #   maxiter = 100,
  #   tol = 1e-6,
  #   trace = F,
  #   plot = F
  # )
  # 
  # message("Computing correlation")
  # 
  # corfit = limma::duplicateCorrelation(all_voom_1, design, block = patient_info)
  # 
  # message("voom with quality weights")
  # 
  # all_voom_2 = limma::voomWithQualityWeights(
  #   dge_tmm,
  #   design,
  #   normalize.method = "none",
  #   method = "genebygene",
  #   maxiter = 100,
  #   tol = 1e-6,
  #   block = patient_info,
  #   correlation = corfit$consensus,
  #   trace = F,
  #   plot = F
  # )
  # 
  # fit_noBayes = lmFit(all_voom_2,
  #                     design,
  #                     block = patient_info,
  #                     correlation = corfit$consensus
  # )
  # 
  # fit = eBayes(fit_noBayes, trend = TRUE, robust = TRUE)
  
  y <- voom(dge_tmm, design, plot = F)
  fit_noBayes <- lmFit(y, design)
  
  # AnyTimepoint ---------------------------------------------------------------
  
  pe_v_control_cont = makeContrasts(
    condTimePreeclampsiaT1 - condTimeControlT1,
    condTimePreeclampsiaT2 - condTimeControlT2,
    condTimePreeclampsiaT3 - condTimeControlT3,
    levels = design
  )
  
  
  fit_pe_v_control = contrasts.fit(fit_noBayes, pe_v_control_cont)
  fit_pe_v_control = eBayes(fit_pe_v_control)
  
  
  return(fit_pe_v_control)
  
}