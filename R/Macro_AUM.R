ROC_AUM_macro<-function(pred_tensor,label_tensor){
  
  if ((pred_tensor$ndim)==1  ) {
    pred_tensor2 <- torch::torch_stack(
      list(1 - pred_tensor, pred_tensor),
      dim = 2
    )
    n_class <- 2
    
  } else {
    if(pred_tensor$size(2)==1){
      pred_tensor2 <-torch::torch_cat(list(1 - pred_tensor, pred_tensor),
                                      dim = 2)
      n_class <- 2
    }
    else{
      n_class <- pred_tensor$size(2)
      pred_tensor2 <-pred_tensor
    }

  }
  one_hot_labels = torch::nnf_one_hot(label_tensor, num_classes=n_class) 
  is_positive = one_hot_labels
  is_negative =1-one_hot_labels
  fn_diff = -is_positive
  fp_diff = is_negative
  
  thresh_tensor = -pred_tensor2
  fn_denom = is_positive$sum(dim = 1)$clamp(min=1)
  fp_denom = is_negative$sum(dim = 1)$clamp(min=1)
  sorted_indices = torch::torch_argsort(thresh_tensor, dim = 1)
  sorted_fp_cum = torch::torch_gather(fp_diff, dim=1, index=sorted_indices)$cumsum(1)/fp_denom

  sorted_fn_cum = -torch::torch_gather(fn_diff, dim=1, index=sorted_indices)$flip(1)$cumsum(1)$flip(1)/fn_denom
  sorted_thresh = torch::torch_gather(thresh_tensor, dim=1, index=sorted_indices)
  zeros_vec=torch::torch_zeros(1,n_class)
  FPR = torch::torch_cat(c(zeros_vec, sorted_fp_cum))
  FNR = torch::torch_cat(c(sorted_fn_cum, zeros_vec))
  roc<- list(
    FPR_all_classes= FPR,
    FNR_all_classes= FNR,
    TPR_all_classes= 1 - FNR,
    "min(FPR,FNR)"= torch::torch_minimum(FPR, FNR),
    min_constant = torch::torch_cat(c(-torch::torch_ones(1,n_class), sorted_thresh)),
    max_constant = torch::torch_cat(c(sorted_thresh, zeros_vec))
  )
  label_int <- label_tensor$to(dtype = torch::torch_int())
  actual_n_classes=torch::torch_bincount(label_int)$size(1)
  min_FPR_FNR = roc[["min(FPR,FNR)"]][2:-2,]
  constant_diff = roc$min_constant[2:N,]$diff(dim=1)
  sum = torch::torch_sum(min_FPR_FNR * constant_diff,dim=1)
  mean=torch::torch_sum(sum)/actual_n_classes
}