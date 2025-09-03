test_that("micro_auc: basic properties and small known cases", {
  skip_on_cran()

  ##  Case 1: perfect 3-class separation  macro AUC = 1
  labels <- torch::torch_tensor(c(1L, 2L, 3L, 1L, 2L, 3L), dtype = torch::torch_long())
  P <- matrix(
    c(
      0.9, 0.05, 0.05,  # class 1
      0.05, 0.9, 0.05,  # class 2
      0.05, 0.05, 0.9,  # class 3
      0.85, 0.10, 0.05, # class 1
      0.02, 0.90, 0.08, # class 2
      0.05, 0.10, 0.85  # class 3
    ),
    ncol = 3, byrow = TRUE
  )
  pred <- torch::torch_tensor(P, dtype = torch::torch_float())

  auc_pkg <- ROC_AUC_macro(pred, labels)$item()

  expect_true(is.numeric(auc_pkg))
  expect_equal(auc_pkg, 1.0, tolerance = 1e-6)

  ## Case 2: random scores baseline around 0.5
  torch::torch_manual_seed(1)
  N <- 300L; K <- 4L
  labels2 <- torch::torch_tensor(sample.int(K, N, replace = TRUE) , dtype = torch::torch_long())
  logits2 <- torch::torch_randn(c(N, K))
  probs2 <- torch::nnf_softmax(logits2, dim = 2)
  auc_pkg2 <- ROC_AUC_macro(probs2, labels2)$item()

  expect_true(auc_pkg2 >= 0 && auc_pkg2 <= 1)
  # Random baseline should be ~0.5; allow some slack with many classes.
  expect_true(abs(auc_pkg2 - 0.5) < 0.1)

})
test_that("macro_aum: basic properties and small known cases", {
  skip_on_cran()

  ##  Case 1: perfect 3-class separation  macro AUC = 1
  labels <- torch::torch_tensor(c(1L, 2L, 3L, 1L, 2L, 3L), dtype = torch::torch_long())
  P <- matrix(
    c(
      0.9, 0.05, 0.05,  # class 1
      0.05, 0.9, 0.05,  # class 2
      0.05, 0.05, 0.9,  # class 3
      0.85, 0.10, 0.05, # class 1
      0.02, 0.90, 0.08, # class 2
      0.05, 0.10, 0.85  # class 3
    ),
    ncol = 3, byrow = TRUE
  )
  pred <- torch::torch_tensor(P, dtype = torch::torch_float())

  aum_pkg <- ROC_AUM_macro(pred, labels)$item()

  expect_true(is.numeric(aum_pkg))
  expect_equal(aum_pkg, 0, tolerance = 1e-6)

})
