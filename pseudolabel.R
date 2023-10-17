
source("rlib/pseudolabel_algorithms.R")

# Load -------------------------------------------------------------------------

set.seed(2023-10-06)
stars <- list.files("data/stars", "prepared-", full.names = TRUE) |> 
  lapply(read.csv, check.names=FALSE)

# Self-train -------------------------------------------------------------------

# Function for doing cross-validation
selftrain_cv <- function(model, predict_probs) {
  lapply(seq_along(stars), function(i) {
    train <- do.call(rbind, stars[-i])
    train <- split(train, train$`car-0.01`)
    
    test <- stars[[i]]
    
    cat("\n")
    selftrain_alpha(
      class~u+g+r+i+z, 
      model,
      predict_probs,
      train[[2]], 
      train[[1]], 
      test
    )
  })
}


cat("\n\n# Self-training with multinomial classifier ------- \n")
res <- selftrain_cv(
  model = \(...) nnet::multinom(..., trace=FALSE),
  predict_probs = \(m, df) predict(m, type = "probs", newdata = df)
)
sapply(res, \(x) 100*x[["acc_diff"]]) |> mean() # Average accuracy change


cat("\n\n# Self-training with QDA classifier ------- \n")
res <- selftrain_cv(
  model = \(...) MASS::qda(...),
  predict_probs = \(m, df) predict(m, newdata = df)$posterior
)
sapply(res, \(x) 100*x[["acc_diff"]]) |> mean() # Average accuracy change


cat("\n\n# Self-training with naive bayes classifier ------- \n")
res <- selftrain_cv(
  model = \(...) naivebayes::naive_bayes(...),
  predict_probs = \(m, df) predict(m, newdata = df, type = "prob")
)
sapply(res, \(x) 100*x[["acc_diff"]]) |> mean() # Average accuracy change

# # SVM IS VERY SLOW! Commenting it out for now. It's similar to the others :/
# cat("\n\n# Self-training with SVM classifier ------- \n")
# res <- selftrain_cv(
#   model = \(...) kernlab::ksvm(..., prob.model=TRUE),
#   predict_probs = \(m, df) kernlab::predict(m, df, type="probabilities")
# )
# sapply(res, \(x) 100*x[["acc_diff"]]) |> mean() # Average accuracy change



# Co-train ---------------------------------------------------------------------
# Function for doing cross-validation
cotrain_cv <- function(model1, predict_probs1, model2, predict_probs2) {
  lapply(seq_along(stars), function(i) {
    train <- do.call(rbind, stars[-i])
    train <- split(train, train$`car-0.01`)
    
    test <- stars[[i]]
    
    cat("\n")
    cotrain_alpha(
      class~u+g+r+i+z, 
      model1, predict_probs1, model2, predict_probs2,
      train[[2]], 
      train[[1]], 
      test,
      epochs = 5
    )
  })
}

cat("\n\n# Co-training with multinomial/QDA classifiers ------- \n")
res <- cotrain_cv(
  model1 = \(...) nnet::multinom(..., trace=FALSE),
  predict_probs1 = \(m, df) predict(m, type = "probs", newdata = df),
  
  model2 = \(...) MASS::qda(...),
  predict_probs2 = \(m, df) predict(m, newdata = df)$posterior
)
sapply(res, \(x) 100*x[["acc_diff"]]) |> mean() # Average accuracy change


cat("\n\n# Co-training with LDA/QDA classifiers ------- \n")
res <- cotrain_cv(
  model1 = \(...) MASS::lda(...),
  predict_probs1 = \(m, df) predict(m, newdata = df)$posterior,
  
  model2 = \(...) MASS::qda(...),
  predict_probs2 = \(m, df) predict(m, newdata = df)$posterior
)
sapply(res, \(x) 100*x[["acc_diff"]]) |> mean() # Average accuracy change

# # SVM IS VERY SLOW! Commenting it out for now. It's similar to the others :/
# cat("\n\n# Co-training with SVM/QDA classifiers ------- \n")
# cotrain_cv(
#   model1 = \(...) kernlab::ksvm(..., prob.model=TRUE),
#   predict_probs1 = \(m, df) kernlab::predict(m, df, type="probabilities"),
#   
#   model2 = \(...) MASS::qda(...),
#   predict_probs2 = \(m, df) predict(m, newdata = df)$posterior
#   
# )

