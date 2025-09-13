#' Compute multi-class ROC AUC micro averaged
#'
#' This function computes the multi class ROC AUC using OvR approach and micro
#' averaging. It assumes that all the inputs are torch tensors and labels are
#' in [1,K] with K being the number of classes.
#'
#' @param pred_tensor output of the model assuming it is of dimension NxK
#'  (or Nx1 for binary classification)
#' @param label_tensor true labels , tensor of length N
#' @return ROC AUC macro averaged
#'
#'
#' @examplesIf torch::torch_is_installed()
#' \dontrun{
#' # Small example with 3 classes and 10 samples
#' labels = torch::torch_randint(1, 4, size = 10, dtype = torch::torch_long())
#' Draw_ROC_curve_micro(torch::torch_randn(c(10, 3)), labels)
#' }
#' @export
ROC_AUC_micro<-function(pred_tensor,label_tensor){
  n_class=pred_tensor$size(2)
  one_hot_labels = torch::nnf_one_hot(label_tensor, num_classes=n_class)
  is_positive = one_hot_labels
  is_negative =1-one_hot_labels
  fn_diff = -is_positive$flatten()
  fp_diff = is_negative$flatten()
  thresh_tensor = -pred_tensor$flatten()
  fn_denom = is_positive$sum()
  fp_denom = is_negative$sum()
  sorted_indices = torch::torch_argsort(thresh_tensor)
  sorted_fp_cum = fp_diff[sorted_indices]$cumsum(dim=1) / fp_denom
  sorted_fn_cum = -fn_diff[sorted_indices]$flip(1)$cumsum(dim=1)$flip(1) / fn_denom

  sorted_thresh = thresh_tensor[sorted_indices]
  sorted_is_diff = sorted_thresh$diff() != 0
  sorted_fp_end = torch::torch_cat(c(sorted_is_diff, torch::torch_tensor(TRUE)))
  sorted_fn_end = torch::torch_cat(c(torch::torch_tensor(TRUE), sorted_is_diff))

  uniq_thresh = sorted_thresh[sorted_fp_end]
  uniq_fp_after = sorted_fp_cum[sorted_fp_end]
  uniq_fn_before = sorted_fn_cum[sorted_fn_end]

  FPR = torch::torch_cat(c(torch::torch_tensor(0.0), uniq_fp_after))
  FNR = torch::torch_cat(c(uniq_fn_before, torch::torch_tensor(0.0)))
  roc <- list(
    FPR=FPR,
    FNR=FNR,
    TPR=1 - FNR,
    "min(FPR,FNR)"=torch::torch_minimum(FPR, FNR),
    min_constant=torch::torch_cat(c(torch::torch_tensor(-1), uniq_thresh)),
    max_constant=torch::torch_cat(c(uniq_thresh, torch::torch_tensor(0))))
  FPR_diff = roc$FPR[2:-1]-roc$FPR[1:-2]
  TPR_sum = roc$TPR[2:-1]+roc$TPR[1:-2]
  return(torch::torch_sum(FPR_diff*TPR_sum/2.0))
}
