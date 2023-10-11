
# Testing cotraining

source("rlib/pseudolabel_algorithms.R")

N_LABELED <- 25

ts_idx <- sample(1:nrow(iris), nrow(iris)/5)
iris_ts <- iris[ts_idx, ]
iris_tr <- iris[-ts_idx, ]
iris_tr_l_idx <- sample(1:nrow(iris_tr), N_LABELED)
iris_tr_l <- iris_tr[iris_tr_l_idx, ]
iris_tr_u <- iris_tr[-iris_tr_l_idx, ]

selftrain_alpha(
  Species~.,
  model = \(...) nnet::multinom(..., trace=FALSE),
  predict_probs = \(m, df) predict(m, type = "probs", newdata = df),
  iris_tr_l, iris_tr_u, iris_ts, 
  alpha=0.9
)

selftrain_n(
  Species~., 
  model = \(...) nnet::multinom(..., trace=FALSE),
  predict_probs = \(m, df) predict(m, type = "probs", newdata = df),
  iris_tr_l, iris_tr_u, iris_ts,
  n=5
)

cotrain_alpha(
  Species~., 
  model1 = \(...) nnet::multinom(..., trace=FALSE),
  predict_probs1 = \(m, df) predict(m, type = "probs", newdata = df),
  model2 = \(...) kernlab::ksvm(..., prob.model=TRUE),
  predict_probs2 = \(m, df) kernlab::predict(m, df, type="probabilities"),
  iris_tr_l, iris_tr_u, iris_ts,
  alpha=0.9
)

cotrain_n(
  Species~., 
  model1 = \(...) nnet::multinom(..., trace=FALSE),
  predict_probs1 = \(m, df) predict(m, type = "probs", newdata = df),
  model2 = \(...) kernlab::ksvm(..., prob.model=TRUE),
  predict_probs2 = \(m, df) kernlab::predict(m, df, type="probabilities"),
  iris_tr_l, iris_tr_u, iris_ts,
  n=5
)
