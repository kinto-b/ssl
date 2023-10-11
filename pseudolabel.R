
source("rlib/pseudolabel_algorithms.R")

# Load -------------------------------------------------------------------------

set.seed(2023-10-06)
stars <- read.csv("data/stars/prepared-train-car-0.01.csv")

stars <- split(stars, is.na(stars$class))
names(stars) <- c("labeled", "unlabeled")

stars$validate <- read.csv("data/stars/prepared-validate.csv")

# Self-train -------------------------------------------------------------------
# Try a self-training with a few different classifiers

cat("\n\n# Self-training with multinomial classifier ------- \n")
selftrain_alpha(
  class~., 
  model = \(...) nnet::multinom(..., trace=FALSE),
  predict_probs = \(m, df) predict(m, type = "probs", newdata = df),
  stars$labeled, stars$unlabeled, stars$validate
)

selftrain_n(
  class~., 
  model = \(...) nnet::multinom(..., trace=FALSE),
  predict_probs = \(m, df) predict(m, type = "probs", newdata = df),
  stars$labeled, stars$unlabeled, stars$validate,
  n=1000
)

cat("\n\n# Self-training with LDA classifier ------- \n")
selftrain(
  class~., 
  model = \(...) MASS::lda(...),
  predict_probs = \(m, df) predict(m, newdata = df)$posterior,
  stars$labeled, stars$unlabeled, stars$validate
)

cat("\n\n# Self-training with QDA classifier ------- \n")
selftrain(
  class~., 
  model = \(...) MASS::qda(...),
  predict_probs = \(m, df) predict(m, newdata = df)$posterior,
  stars$labeled, stars$unlabeled, stars$validate
)

cat("\n\n# Self-training with naive bayes classifier ------- \n")
selftrain(
  class~., 
  model = \(...) naivebayes::naive_bayes(...),
  predict_probs = \(m, df) predict(m, newdata = df, type = "prob"),
  stars$labeled, stars$unlabeled, stars$validate
)

# # SVM IS VERY SLOW! Commenting it out for now. It's similar to the others :/
# cat("\n\n# Self-training with SVM classifier ------- \n")
# selftrain(
#   class~., 
#   model = \(...) kernlab::ksvm(..., prob.model=TRUE),
#   predict_probs = \(m, df) kernlab::predict(m, df, type="probabilities"),
#   stars$labeled, stars$unlabeled, stars$validate
# )


# Co-train ---------------------------------------------------------------------
# Try co-training with a few different pairs

cat("\n\n# Co-training with multinomial/QDA classifiers ------- \n")
cotrain(
  class~., 
  model1 = \(...) nnet::multinom(..., trace=FALSE),
  predict_probs1 = \(m, df) predict(m, type = "probs", newdata = df),
  
  model2 = \(...) MASS::qda(...),
  predict_probs2 = \(m, df) predict(m, newdata = df)$posterior,
  
  stars$labeled, stars$unlabeled, stars$validate,
  epochs = 5
)

cat("\n\n# Co-training with LDA/QDA classifiers ------- \n")
cotrain(
  class~., 
  model1 = \(...) MASS::lda(...),
  predict_probs1 = \(m, df) predict(m, newdata = df)$posterior,
  
  model2 = \(...) MASS::qda(...),
  predict_probs2 = \(m, df) predict(m, newdata = df)$posterior,
  
  stars$labeled, stars$unlabeled, stars$validate,
  epochs = 5
)

# # SVM IS VERY SLOW! Commenting it out for now. It's similar to the others :/
# cat("\n\n# Co-training with SVM/QDA classifiers ------- \n")
# cotrain(
#   class~., 
#   model1 = \(...) kernlab::ksvm(..., prob.model=TRUE),
#   predict_probs1 = \(m, df) kernlab::predict(m, df, type="probabilities"),
#   
#   model2 = \(...) MASS::qda(...),
#   predict_probs2 = \(m, df) predict(m, newdata = df)$posterior,
#   
#   stars$labeled, stars$unlabeled, stars$validate,
#   epochs = 5
# )

