#' @importFrom ggplot2 ggplot aes geom_line labs theme_minimal
#' @import data.table
NULL
#' Draws multi-class ROC curve macro 
#'
#' This function draws K ROC curves  using OvR approach, each time 
#' considering one class as the positive class.  
#' It assumes that all the inputs are torch tensors and labels are 
#' in [1,K] with K being the number of classes.
#' 
#' @param pred_tensor output of the model assuming it is of dimension NxK
#'  (or Nx1 for binary classification)
#' @param label_tensor true labels , tensor of length N
#' @return K ROC curves 
#' 
#'
#' @examples
#' #Small example with 3 classes and 10 samples
#' labels=torch::torch_randint(1, 4, size = 10,dtype = torch::torch_long())
#' Draw_ROC_curve_macro(torch::torch_randn(c(10, 3)),labels)
#' @export
Draw_ROC_curve_macro<-function(pred_tensor, label_tensor){
  n_class=pred_tensor$size(2)
  one_hot_labels = torch::nnf_one_hot(label_tensor, num_classes = n_class)
  is_positive = one_hot_labels
  is_negative =1-one_hot_labels
  fn_diff = -is_positive
  fp_diff = is_negative
  thresh_tensor = -pred_tensor
  fn_denom = is_positive$sum(dim = 1)
  fp_denom = is_negative$sum(dim = 1)
  sorted_indices = torch::torch_argsort(thresh_tensor, dim = 1)
  sorted_fp_cum = torch::torch_gather(fp_diff, dim=1, index=sorted_indices)$cumsum(1)/fp_denom
  sorted_fn_cum = -torch::torch_gather(fn_diff, dim=1, index=sorted_indices)$flip(1)$cumsum(1)$flip(1)/fn_denom
  sorted_thresh = torch::torch_gather(thresh_tensor, dim=1, index=sorted_indices)
  zeros_vec=torch::torch_zeros(1,n_class)
  FPR = torch::torch_cat(c(zeros_vec, sorted_fp_cum))
  FNR = torch::torch_cat(c(sorted_fn_cum, zeros_vec))
  TPR= 1 - FNR
  
  fpr_mat <- torch::as_array(FPR)  
  tpr_mat <- torch::as_array(TPR)
  n_points <- dim(fpr_mat)[1]
  fpr = as.vector(fpr_mat)
  tpr = as.vector(tpr_mat)
  class = rep(1:n_class, each = n_points)
  roc_dt_macro <- data.table(
    fpr = fpr,
    tpr = tpr,
    class = class
  )
  ggplot(roc_dt_macro, aes(x = fpr, y = tpr, color = factor(class))) +
    geom_line() +
    labs(title = "ROC Curves macro", x = "FPR", y = "TPR", color = "Class") +
    theme_minimal()
}