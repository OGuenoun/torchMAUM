Draw_ROC_curve_micro <- function(pred_tensor, label_tensor){
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
  
  FPR = as.array(torch::torch_cat(c(torch::torch_tensor(0.0), uniq_fp_after)))
  TPR = as.array(1-torch::torch_cat(c(uniq_fn_before, torch::torch_tensor(0.0))))
  
  dt=data.table::data.table(
    FPR = FPR,
    TPR = TPR
    )
  ggplot(dt, aes(x = FPR, y = TPR)) +
    geom_line(color = "blue", size = 1) +
    theme_minimal() +
    labs(
      title = "Roc curve using micro average",
      x = "FPR",
      y = "TPR"
    ) 
}