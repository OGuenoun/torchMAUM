ROC_AUM_micro<-function(pred_tensor,label_tensor,counts=NULL){
  n_class=pred_tensor$size(2)
  one_hot_labels = torch::nnf_one_hot(label_tensor, num_classes=n_class) 
  if(!is.null(counts)){
    stopifnot(length(counts) == n_class)
    Pweights <- 1 / (counts + 1e-8)
    Pweights <- Pweights / Pweights$sum()
    Nweights <-1/(counts$sum()-counts+1e-8)
    Nweights <-Nweights/ Nweights$sum()
  }
  else{
    Pweights<-1
    Nweights<-1
  }
  is_positive = one_hot_labels*Pweights
  is_negative =(1-one_hot_labels)*Nweights
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
  min_FPR_FNR = roc[["min(FPR,FNR)"]][2:-2]
  constant_diff = roc$min_constant[2:N]$diff()
  return(torch::torch_sum(min_FPR_FNR * constant_diff))
}
ROC_AUM_micro(four_pred,four_labels,counts=counts)
