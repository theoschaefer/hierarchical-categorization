distance_gcm <- function(i, tbl_x, r) {
  N <- nrow(tbl_x)
  m <- matrix(unlist(rep(tbl_x[i, c("x1", "x2")], N)), N, 2, byrow = TRUE)
  colnames(m) <- c("x1", "x2")
  tbl_single <- as_tibble(m)
  ((tbl_single$x1 - tbl_x$x1)^r +
      (tbl_single$x2 - tbl_x$x2)^r
  )^(1/r)
}

distance_1d <- function(i, tbl_x, c) {
  N <- nrow(tbl_x)
  m <- matrix(unlist(rep(tbl_x[i, c], N)), N, 1, byrow = TRUE)
  colnames(m) <- c
  tbl_single <- as_tibble(m)
  abs(tbl_single[, c] - tbl_x[, c])
}

write_gcm_stan_file <- function() {
  write_stan_file("
data {
  int n_stim;
  int n_cat;
  array[n_stim] int n_trials; // n trials
  array[n_stim] int n_correct; // n correct categorization responses
  array[n_stim] int n_correct_predict; // n correct categorization responses on test set
  array[n_stim] int n_trials_per_item;
  array[n_stim] int cat; // actual category for a given stimulus
  array[n_stim, n_stim] real<lower=0> d1;
  array[n_stim, n_stim] real<lower=0> d2;
  
}


parameters {
  real <lower=0> c;
  //real <lower=0,upper=1> w;
  simplex[n_cat] bs;
}

transformed parameters {
  array[n_stim, n_stim] real <lower=0,upper=1> s;
  array[n_stim, n_cat] real <lower=0> sumsim;
  array[n_stim] real <lower=0,upper=1> theta;

  
  // Similarities
  for (i in 1:n_stim){
    for (k in 1:n_cat) {
      sumsim[i, k] = 0;
    }
    for (j in 1:n_stim){
      //s[i, j] = exp(-square(c)*(w*square(d1[i, j])+(1-w)*square(d2[i, j])));
      s[i, j] = exp(-square(c)*(.5*square(d1[i, j])+(.5)*square(d2[i, j])));
      sumsim[i, cat[j]] = sumsim[i, cat[j]] + s[i,j] * bs[cat[j]];
    }
    theta[i] = sumsim[i, cat[i]] / sum(sumsim[i, ]);
  }
}

model {
  n_correct ~ binomial(n_trials, theta);
  c ~ uniform(0, 10);
  //w ~ beta(1, 1);

}

generated quantities {
  array[n_stim] real log_lik_pred;

  for (n in 1:n_stim) {
  log_lik_pred[n] = binomial_lpmf(n_correct_predict[n] | n_trials_per_item[n], theta[n]);
  }
}

")
}


write_gcm_stan_file_predict <- function() {
  write_stan_file("
data {
  int n_stim;
  int n_cat;
  array[n_stim] int n_trials; // n trials
  array[n_stim] int n_correct; // n correct categorization responses
  array[n_stim] int cat; // actual category for a given stimulus
  array[n_stim, n_stim] real<lower=0> d1;
  array[n_stim, n_stim] real<lower=0> d2;
  
  int n_stim_predict;
  array[n_stim_predict] int n_trials_predict; // n trials on test set
  array[n_stim_predict] int n_correct_predict; // n correct categorization responses on test set
  array[n_stim_predict] int cat_predict; // actual category for a given stimulus
  array[n_stim_predict, n_stim_predict] real<lower=0> d1_predict;
  array[n_stim_predict, n_stim_predict] real<lower=0> d2_predict;
  
}


parameters {
  real <lower=0> c;
  //real <lower=0,upper=1> w;
  simplex[n_cat] bs;
}

transformed parameters {
  array[n_stim, n_stim] real <lower=0,upper=1> s;
  array[n_stim, n_cat] real <lower=0> sumsim;
  array[n_stim] real <lower=0,upper=1> theta;

  
  // Similarities
  for (i in 1:n_stim){
    for (k in 1:n_cat) {
      sumsim[i, k] = 0;
    }
    for (j in 1:n_stim){
      //s[i, j] = exp(-square(c)*(w*square(d1[i, j])+(1-w)*square(d2[i, j])));
      s[i, j] = exp(-square(c)*(.5*square(d1[i, j])+(.5)*square(d2[i, j])));
      sumsim[i, cat[j]] = sumsim[i, cat[j]] + s[i,j] * bs[cat[j]];
    }
    theta[i] = sumsim[i, cat[i]] / sum(sumsim[i, ]);
  }
}

model {
  n_correct ~ binomial(n_trials, theta);
  c ~ uniform(0, 10);
  //w ~ beta(1, 1);

}

generated quantities {
  array[n_stim_predict] real log_lik_pred;
  array[n_stim_predict, n_stim_predict] real <lower=0,upper=1> s_predict;
  array[n_stim_predict, n_cat] real <lower=0> sumsim_predict;
  array[n_stim_predict] real <lower=0,upper=1> theta_predict;


  // Similarities
  for (i in 1:n_stim_predict){
    for (k in 1:n_cat) {
      sumsim_predict[i, k] = 0;
    }
    for (j in 1:n_stim_predict){
      //s_predict[i, j] = exp(-square(c)*(w*square(d1_predict[i, j])+(1-w)*square(d2_predict[i, j])));
      s_predict[i, j] = exp(-square(c)*(.5*square(d1_predict[i, j])+(.5)*square(d2_predict[i, j])));
      sumsim_predict[i, cat_predict[j]] = sumsim_predict[i, cat_predict[j]] + s_predict[i,j] * bs[cat_predict[j]];
    }
    theta_predict[i] = sumsim_predict[i, cat_predict[i]] / sum(sumsim_predict[i, ]);
    log_lik_pred[i] = binomial_lpmf(n_correct_predict[i] | n_trials_predict[i], theta_predict[i]);
  }

}

")
}

write_gaussian_naive_bayes_stan <- function() {
  write_stan_file("
  
data {
 int D; //number of dimensions
 int K; //number of categories
 int N; //number of data
 int n_stim; // nr of different stimuli
 array[n_stim] int cat_true; // true category for a stimulus
 array[N] int cat; //category response for a stimulus
 matrix[N, D] y;
 matrix[n_stim, D] y_unique;
 array[n_stim] int n_correct_predict; // n correct categorization responses on test set
 array[n_stim] int n_trials_per_item;
}

parameters {
 ordered[K] mu1; //category means d1
 ordered[K] mu2; //category means d2
 array[D, K] real<lower=0> sigma; //variance
}

transformed parameters {
 vector[n_stim] theta;
 
 for (n in 1:n_stim){
   vector[K] LL1_unique;
   vector[K] LL2_unique;
   for (k in 1:K) {
     LL1_unique[k] = normal_lpdf(y_unique[n, 1] | mu1[k], sigma[1, k]);
     LL2_unique[k] = normal_lpdf(y_unique[n, 2] | sort_desc(mu2)[k], sigma[2, k]);
   }
   theta[n] = exp(LL1_unique[cat_true[n]] + LL2_unique[cat_true[n]]) / (
   exp(LL1_unique[1] + LL2_unique[1]) + exp(LL1_unique[2] + LL2_unique[2]) + exp(LL1_unique[3] + LL2_unique[3])
   );
 }
}

model {
  
 for(k in 1:K){
   for (d in 1:D) {
     sigma[d, k] ~ uniform(0.1, 5);
   }
 }
 mu1[1] ~ normal(-1.5, .5);
 mu1[2] ~ normal(0, .5);
 mu1[3] ~ normal(1.5, .5);
 mu2[1] ~ normal(-1.5, .5);
 mu2[2] ~ normal(0, .5);
 mu2[3] ~ normal(1.5, .5);

 for (n in 1:N){
 vector[K] LL1;
 vector[K] LL2;
   for (k in 1:K) {
     LL1[k] = normal_lpdf(y[n, 1] | mu1[k], sigma[1, k]);
     LL2[k] = normal_lpdf(y[n, 2] | sort_desc(mu2)[k], sigma[2, k]);
   }
  target += LL1[cat[n]] + LL2[cat[n]];
 }
}

generated quantities {
  array[n_stim] real log_lik_pred;

  for (n in 1:n_stim) {
  log_lik_pred[n] = binomial_lpmf(n_correct_predict[n] | n_trials_per_item[n], theta[n]);
  }
}


")
}


write_gaussian_naive_bayes_stan_recovery <- function() {
  write_stan_file("
  
data {
 int D; //number of dimensions
 int K; //number of categories
 int N; //number of data
 int n_stim; // nr of different stimuli
 array[n_stim] int cat_true; // true category for a stimulus
 array[N] int cat; //category response for a stimulus
 matrix[N, D] y;
 matrix[n_stim, D] y_unique;
 array[n_stim] int n_correct_predict; // n correct categorization responses on test set
 array[n_stim] int n_trials_per_item;
  
}

parameters {
 array[D] ordered[K] mu; //category means
 array[D, K] real<lower=0> sigma; //variance
}

transformed parameters {
 vector[n_stim] theta;
 
 for (n in 1:n_stim){
   vector[K] LL1_unique;
   vector[K] LL2_unique;
   for (k in 1:K) {
     LL1_unique[k] = normal_lpdf(y_unique[n, 1] | mu[1][k], sigma[1, k]);
     LL2_unique[k] = normal_lpdf(y_unique[n, 2] | mu[2][k], sigma[2, k]);
   }
   theta[n] = exp(LL1_unique[cat_true[n]] + LL2_unique[cat_true[n]]) / (
   exp(LL1_unique[1] + LL2_unique[1]) + exp(LL1_unique[2] + LL2_unique[2]) + exp(LL1_unique[3] + LL2_unique[3])
   );
 }
}

model {
  
 for(k in 1:K){
   for (d in 1:D) {
     sigma[d, k] ~ uniform(0.1, 5);
   }
 }
 mu[1][1] ~ normal(-1.5, .5);
 mu[1][2] ~ normal(0, .5);
 mu[1][3] ~ normal(1.5, .5);
 mu[2][1] ~ normal(-1.5, .5);
 mu[2][2] ~ normal(0, .5);
 mu[2][3] ~ normal(1.5, .5);

 for (n in 1:N){
 vector[K] LL1;
 vector[K] LL2;
   for (k in 1:K) {
     LL1[k] = normal_lpdf(y[n, 1] | mu[1][k], sigma[1, k]);
     LL2[k] = normal_lpdf(y[n, 2] | mu[2][k], sigma[2, k]);
   }
  target += LL1[cat[n]] + LL2[cat[n]];
 }
}


generated quantities {
  array[n_stim] real log_lik_pred;

  for (n in 1:n_stim) {
  log_lik_pred[n] = binomial_lpmf(n_correct_predict[n] | n_trials_per_item[n], theta[n]);
  }
}

")
}




write_gaussian_multi_bayes_stan <- function() {
  write_stan_file("
  
data {
 int D; //number of dimensions
 int K; //number of categories
 int N; //number of data
 int n_stim; // nr of different stimuli
 array[n_stim] int cat_true; // true category for a stimulus
 array[N] int cat; //category response for a stimulus
 matrix[N, D] y;
 matrix[n_stim, D] y_unique;
 array[n_stim] int n_correct_predict; // n correct categorization responses on test set
 array[n_stim] int n_trials_per_item;
}

parameters {
 array[K] ordered[D] mu; //category means
 array[K] cholesky_factor_corr[D] L; //cholesky factor of covariance
 // Sqrt of variances for each variate
 array[K] vector<lower=0>[D] L_std;
}

transformed parameters {
 vector[n_stim] theta;
 
 for (n in 1:n_stim){
   vector[K] LL_unique;
   for (k in 1:K) {
     matrix[D, D] L_Sigma_unique = diag_pre_multiply(L_std[k], L[k]);
     LL_unique[k] = multi_normal_cholesky_lpdf(y_unique[n] | mu[k], L_Sigma_unique);
   }
   theta[n] = exp(LL_unique[cat_true[n]] - log_sum_exp(LL_unique));
   //theta[n] = exp(LL_unique[cat_true[n]]) / (exp(LL_unique[1]) + exp(LL_unique[2]) + exp(LL_unique[3]));
 }
}

model {
  
 for(k in 1:K){
   mu[k] ~ normal(0, 1);
   L[k] ~ lkj_corr_cholesky(1);
   L_std[k] ~ normal(0, 2.5);
 }

 for (n in 1:N){
 vector[K] LL;
 for (k in 1:K) {
     matrix[D, D] L_Sigma = diag_pre_multiply(L_std[k], L[k]);
     LL[k] = multi_normal_cholesky_lpdf(y[n] | mu[k], L_Sigma); 
   }
  target += LL[cat[n]];
 }
}

generated quantities {
 array[K] corr_matrix[D] Sigma;
 array[n_stim] real log_lik_pred;
 
 for (k in 1:K){
   Sigma[k] = multiply_lower_tri_self_transpose(L[k]);
 }
 
 for (n in 1:n_stim) {
   log_lik_pred[n] = binomial_lpmf(n_correct_predict[n] | n_trials_per_item[n], theta[n]);
 }
}

")
}


gcm_distances_similarities <- function(tbl_df, l_params) {
  #' gcm distances and similarities for all observations
  #' 
  #' @description compute pairwise distances and similarities between
  #' all observations (i.e., rows)
  #' 
  #' @param tbl_df tbl df with x1 and x2 features
  #' @param l_params gcm model parameters
  #' @return nested list with distances and similarities
  #'   
  # compute pairwise distances
  m_distances_x1 <- map(1:nrow(tbl_df), distance_1d, tbl_df, "x1") %>% unlist() %>%
    matrix(byrow = TRUE, nrow = nrow(tbl_df), ncol = nrow(tbl_df))
  m_distances_x2 <- map(1:nrow(tbl_df), distance_1d, tbl_df, "x2") %>% unlist() %>%
    matrix(byrow = TRUE, nrow = nrow(tbl_df), ncol = nrow(tbl_df))
  m_similarities <- exp(-l_params[["c"]]^2*(l_params[["w"]]*m_distances_x1^2+(1-l_params[["w"]])*m_distances_x2^2))
  
  l_dist_sim <- list(
    m_distances_x1 = m_distances_x1,
    m_distances_x2 = m_distances_x2,
    m_similarities = m_similarities
  )
  
  return(l_dist_sim)
}



gcm_response_probabilities <- function(i, tbl_df, m_sims, l_params) {
  #' compute gcm response probabilities given pairwise similarities
  #' 
  #' @description computes summed similarity of within category items
  #' divided by summed similarity of all items
  #' 
  #' @param i idx of the row, for which to compute the response probability
  #' @param tbl_df tbl df with x1 and x2 features and category column
  #' @param m_sims nxn matrix with pairwise similarities
  #' @param l_params list with model parameters
  #' @return dbl with response probability of being correct
  #' 
  sum_sim_category <- function(i, tbl_df) {
    cats_distinct <- unique(tbl_df$category)
    l_mask <- map(cats_distinct, ~ tbl_df$category == .x)
    map_dbl(l_mask, ~ sum(m_sims[i, ][.x]))
  }
  
  sum_sim_categories <- sum_sim_category(i, tbl_df)
  sum_sim_biased <- sum_sim_categories * l_params[["b"]]
  
  sum_sim_biased[tbl_df$category[i]] / sum(sum_sim_biased)
}


my_rbinom <- function(n_trials, prob_correct_true) {
  rbinom(n = 1, size = n_trials, prob = prob_correct_true)
}




aggregate_by_stimulus_and_response <- function(tbl_stim_id, tbl_df) {
  #' aggregate responses by participant, stimulus id, category, and response
  #' 
  #' @description make sure categories not responded to are filled with 0s
  #' 
  #' @param tbl_stim_id tbl_df containing all training stim_id with 
  #' associated x values and categories
  #' @param tbl_df tbl df with all category learning training data
  #' @return aggregated tbl df
  #' 
  
  session <- tbl_df$session[1]
  tbl_design <- tbl_stim_id %>% 
    crossing(
      response = unique(tbl_df$response), 
      participant = unique(tbl_df$participant)
    ) %>%
    relocate(stim_id, .before = category)
  
  tbl_train_agg <- tbl_df %>% 
    group_by(participant, session, stim_id, d1i, d2i, d1i_z, d2i_z, category, response) %>%
    summarize(
      n_responses = n()
    )
  
  tbl_ <- tbl_design %>% 
    left_join(
      tbl_train_agg, by = c(
        "participant", "stim_id", "d1i", "d2i", "d1i_z", "d2i_z", "category", "response"
      )
    )
  
  tbl_$n_responses[is.na(tbl_$n_responses)] <- 0
  tbl_$session[is.na(tbl_$session)] <- session
  tbl_ %>% group_by(participant, d1i, d2i) %>%
    mutate(
      n_trials = sum(n_responses),
      prop_responses = n_responses / n_trials
    ) %>%
    ungroup()
}


save_my_png <- function(pl, f_name, vals_size) {
  #' save three plots from the categorization model
  #'
  png(filename = f_name, vals_size[1], vals_size[2], "in", res = 200)
  grid.draw(pl)
  dev.off()
}

bayesian_gcm <- function(tbl_participant, l_stan_params, mod_gcm) {
  #' fit by-participant gcm stan model
  #' 
  #' @description fit stan model, save three plots, and return loo
  #' 
  #' @param tbl_participant by-participant aggregated responses from infpro task
  #' @param l_stan_params list with parameters used for stan model
  #' @param mod_gcm the compiled cmdstanr model
  #' @return loo
  #'
  
  tbl_train <- tbl_participant %>% filter(session == "train")
  tbl_transfer <- tbl_participant %>% filter(session == "transfer")
  participant_sample <- tbl_train$participant[1]
  tbl_gcm_train <- tbl_train %>% 
    filter(category == response) %>% 
    mutate(prop_correct = prop_responses)
  tbl_gcm_transfer <- tbl_transfer %>%
    filter(category == response) %>%
    mutate(prop_correct = prop_responses)
  
  # compute pairwise distances
  m_distances_x1_train <- map(1:nrow(tbl_gcm_train), distance_1d, tbl_gcm_train, "d1i_z") %>% unlist() %>%
    matrix(byrow = TRUE, nrow = nrow(tbl_gcm_train), ncol = nrow(tbl_gcm_train))
  m_distances_x2_train <- map(1:nrow(tbl_gcm_train), distance_1d, tbl_gcm_train, "d2i_z") %>% unlist() %>%
    matrix(byrow = TRUE, nrow = nrow(tbl_gcm_train), ncol = nrow(tbl_gcm_train))
  
  m_distances_x1_transfer <- map(1:nrow(tbl_gcm_transfer), distance_1d, tbl_gcm_transfer, "d1i_z") %>% unlist() %>%
    matrix(byrow = TRUE, nrow = nrow(tbl_gcm_transfer), ncol = nrow(tbl_gcm_transfer))
  m_distances_x2_transfer <- map(1:nrow(tbl_gcm_transfer), distance_1d, tbl_gcm_transfer, "d2i_z") %>% unlist() %>%
    matrix(byrow = TRUE, nrow = nrow(tbl_gcm_transfer), ncol = nrow(tbl_gcm_transfer))
  
  l_data <- list(
    # train
    n_stim = nrow(tbl_gcm_train), 
    n_cat = length(unique(tbl_gcm_train$category)),
    n_trials = tbl_gcm_train$n_trials, 
    n_correct = tbl_gcm_train$n_responses,
    cat = as.numeric(factor(tbl_gcm_train$category, labels = c(1, 2, 3))),
    d1 = m_distances_x1_train, 
    d2 = m_distances_x2_train,
    # transfer / predict
    n_stim_predict = nrow(tbl_gcm_transfer),
    n_trials_predict = tbl_gcm_transfer$n_trials,
    n_correct_predict = tbl_gcm_transfer$n_responses,
    cat_predict = as.numeric(factor(tbl_gcm_transfer$category, labels = c(1, 2, 3))),
    d1_predict = m_distances_x1_transfer, 
    d2_predict = m_distances_x2_transfer
  )
  
  fit_gcm <- mod_gcm$sample(
    data = l_data, chains = l_stan_params$n_chains, 
    iter_sampling = l_stan_params$n_samples, iter_warmup = l_stan_params$n_warmup
  )
  
  file_loc <- str_c("data/infpro_task-cat_beh/models/gcm-model-", participant_sample, ".RDS")
  fit_gcm$save_object(file = file_loc)
  
  loo_gcm <- fit_gcm$loo(variables = "log_lik_pred")
  
  pars_interest <- c("theta_predict", "bs", "c")#, "w")
  pars_interest_no_theta <- c("bs", "c")#, "w")
  tbl_draws <- fit_gcm$draws(variables = pars_interest, format = "df")
  
  names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta_predict")]
  tbl_gcm_transfer$pred_theta <- colMeans(tbl_draws[, names_thetas])
  tbl_gcm_transfer$pred_difference <- tbl_gcm_transfer$pred_theta - tbl_gcm_transfer$prop_responses
  
  
  tbl_summary <- fit_gcm$summary(variables = c("theta", "bs", "c"))
  tbl_summary_nok <- tbl_summary %>% filter(rhat > 1.025 | rhat < 0.975)
  if (nrow(tbl_summary_nok) > 0) {
    stop(str_c(
      "participant = ", participant_sample, "; Rhat for some parameters not ok; ",
      "model can be found under: ", 
    ))
  }
  file_loc <- str_c("data/infpro_task-cat_beh/models/gcm-summary-", participant_sample, ".RDS")
  saveRDS(tbl_summary, file_loc)
  
  idx_no_theta <- map(pars_interest_no_theta, ~ str_detect(tbl_summary$variable, .x)) %>%
    reduce(rbind) %>% colSums()
  tbl_label <- tbl_summary[as.logical(idx_no_theta), ]
  
  tbl_posterior <- tbl_draws %>% 
    select(starts_with(pars_interest_no_theta), .chain) %>%
    rename(chain = .chain) %>%
    pivot_longer(starts_with(pars_interest_no_theta), names_to = "parameter", values_to = "value") %>%
    filter(parameter != "chain")
  tbl_posterior$parameter <- fct_inorder(tbl_posterior$parameter)
  
  pl_thetas <- plot_item_thetas(tbl_gcm_transfer, str_c("GCM; Participant = ", participant_sample))
  pl_posteriors <- plot_posteriors(tbl_posterior, tbl_label)
  pl_pred_uncertainty <- plot_proportion_responses(tbl_gcm_transfer, participant_sample, color_pred_difference = TRUE)
  
  # save plots
  c_names <- function(x, y) str_c("data/infpro_task-cat_beh/model-plots/", x, y, ".png")
  l_pl_names <- map(c("gcm-thetas-", "gcm-posteriors-", "gcm-uncertainty-"), c_names, y = participant_sample)
  l_pl <- list(pl_thetas, pl_posteriors, pl_pred_uncertainty)
  l_vals_size <- list(c(3.5, 3.5), c(8.5, 3), c(7.5, 7.5))
  pwalk(list(l_pl, l_pl_names, l_vals_size), save_my_png)
  
  return(loo_gcm)
}


bayesian_gaussian_naive_bayes <- function(
    tbl_participant, tbl_participant_agg, l_stan_params, mod_gaussian
) {
  #' fit by-participant 1D Gaussian naive bayes stan model
  #' 
  #' @description fit stan model, save three plots, and return loo
  #' 
  #' @param tbl_participant by-participant and by-trial responses from infpro task
  #' @param tbl_participant_agg by-participant aggregated responses from infpro task
  #' @param l_stan_params list with parameters used for stan model
  #' @param mod_gaussian the compiled cmdstanr model
  #' @return loo
  #'
  
  tbl_train <- tbl_participant %>% filter(session == "train")
  tbl_participant_agg <- tbl_participant_agg %>% 
    filter(response == category & session == "transfer") %>%
    mutate(prop_correct = prop_responses)
  participant_sample <- tbl_train$participant[1]
  
  l_data <- list(
    D = 2, K = length(unique(tbl_train$category)),
    N = nrow(tbl_train),
    y = tbl_train[, c("d1i_z", "d2i_z")] %>% as.matrix(),
    cat = tbl_train$response_int,
    cat_true = tbl_participant_agg$response_int,
    n_stim = nrow(tbl_participant_agg),
    y_unique = tbl_participant_agg[, c("d1i_z", "d2i_z")] %>% as.matrix(),
    n_correct_predict = tbl_participant_agg$n_responses,
    n_trials_per_item = tbl_participant_agg$n_trials
  )
  fit_gaussian <- mod_gaussian$sample(
    data = l_data, chains = l_stan_params$n_chains, 
    iter_sampling = l_stan_params$n_samples, iter_warmup = l_stan_params$n_warmup
  )
  file_loc <- str_c(
    "data/infpro_task-cat_beh/models/gaussian-model-", participant_sample, ".RDS"
  )
  fit_gaussian$save_object(file = file_loc)
  
  # loo
  loo_gaussian <- fit_gaussian$loo(variables = "log_lik_pred")
  
  pars_interest <- c("mu1", "mu2", "sigma", "theta")
  pars_interest_no_theta <- c("mu1", "mu2", "sigma")
  tbl_draws <- fit_gaussian$draws(variables = pars_interest, format = "df")
  tbl_draws <- tbl_draws %>% rename(`mu2[1]` = `mu2[3]`, `mu2[3]` = `mu2[1]`)
  
  names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
  tbl_participant_agg$pred_theta <- colMeans(tbl_draws[, names_thetas])
  tbl_participant_agg$pred_difference <- tbl_participant_agg$pred_theta - tbl_participant_agg$prop_responses
  
  
  tbl_summary <- fit_gaussian$summary(variables = pars_interest)
  # rename to correct for reverse ordering in stan model
  old_3 <- tbl_summary$variable == "mu2[3]"
  old_1 <- tbl_summary$variable == "mu2[1]"
  tbl_summary$variable[old_3] <- "mu2[1]"
  tbl_summary$variable[old_1] <- "mu2[3]"
  
  tbl_summary_nok <- tbl_summary %>% filter(rhat > 1.025 | rhat < 0.975)
  if (nrow(tbl_summary_nok) > 0) {
    stop(str_c(
      "participant = ", participant_sample, "; Rhat for some parameters not ok",
      "model can be found under: ", file_loc
    ))
  }
  
  file_loc <- str_c("data/infpro_task-cat_beh/models/gaussian-summary-", participant_sample, ".RDS")
  saveRDS(tbl_summary, file_loc)
  
  idx_no_theta <- map(pars_interest_no_theta, ~ str_detect(tbl_summary$variable, .x)) %>%
    reduce(rbind) %>% colSums()
  tbl_label <- tbl_summary[as.logical(idx_no_theta), ]
  
  tbl_posterior <- tbl_draws %>% 
    dplyr::select(starts_with(pars_interest_no_theta), .chain) %>%
    rename(chain = .chain) %>%
    pivot_longer(starts_with(pars_interest_no_theta), names_to = "parameter", values_to = "value") %>%
    filter(parameter != "chain")
  tbl_posterior$parameter <- fct_inorder(tbl_posterior$parameter)
  
  pl_thetas <- plot_item_thetas(tbl_participant_agg, str_c("Gaussian 1D; Participant = ", participant_sample))
  pl_posteriors <- plot_posteriors(tbl_posterior, tbl_label, n_cols = 3)
  pl_pred_uncertainty <- plot_proportion_responses(tbl_participant_agg, participant_sample, color_pred_difference = TRUE)
  
  # save plots
  c_names <- function(x, y) str_c("data/infpro_task-cat_beh/model-plots/", x, y, ".png")
  l_pl_names <- map(
    c("gaussian-1d-thetas-", "gaussian-1d-posteriors-", "gaussian-1d-uncertainty-"),
    c_names, y = participant_sample
  )
  l_pl <- list(pl_thetas, pl_posteriors, pl_pred_uncertainty)
  l_vals_size <- list(c(3.5, 3.5), c(7.5, 8.5), c(7.5, 7.5))
  pwalk(list(l_pl, l_pl_names, l_vals_size), save_my_png)
  
  return(loo_gaussian)
}


bayesian_gaussian_multi_bayes <- function(
    tbl_participant, tbl_participant_agg, l_stan_params, mod_multi
) {
  #' fit by-participant multivariate 2D Gaussian stan model
  #' 
  #' @description fit stan model, save three plots, and return loo
  #' 
  #' @param tbl_participant by-participant and by-trial responses from infpro task
  #' @param tbl_participant_agg by-participant aggregated responses from infpro task
  #' @param l_stan_params list with parameters used for stan model
  #' @param mod_multi the compiled cmdstanr model
  #' @return loo
  #'
  
  tbl_train <- tbl_participant %>% filter(session == "train")
  tbl_participant_agg <- tbl_participant_agg %>% 
    filter(response == category & session == "transfer") %>%
    mutate(prop_correct = prop_responses)
  participant_sample <- tbl_train$participant[1]
  
  l_data <- list(
    D = 2, K = length(unique(tbl_train$category)),
    N = nrow(tbl_train),
    y = tbl_train[, c("d1i_z", "d2i_z")] %>% as.matrix(),
    cat = tbl_train$response_int,
    cat_true = tbl_participant_agg$response_int,
    n_stim = nrow(tbl_participant_agg),
    y_unique = tbl_participant_agg[, c("d1i_z", "d2i_z")] %>% as.matrix(),
    n_correct_predict = tbl_participant_agg$n_responses,
    n_trials_per_item = tbl_participant_agg$n_trials
  )
  fit_multi <- mod_multi$sample(
    data = l_data, chains = l_stan_params$n_chains, 
    iter_sampling = l_stan_params$n_samples, iter_warmup = l_stan_params$n_warmup
  )
  file_loc <- str_c(
    "data/infpro_task-cat_beh/models/multi-model-", participant_sample, ".RDS"
  )
  fit_multi$save_object(file = file_loc)
  
  # loo
  loo_multi <- fit_multi$loo(variables = "log_lik_pred")
  
  
  pars_interest <- c("mu", "Sigma", "theta")
  tbl_draws <- fit_multi$draws(variables = pars_interest, format = "df")
  names_params <- names(tbl_draws)
  names_sigmas <- names_params[startsWith(names_params, "Sigma")]
  sigmas_keep <- map(c("2,2]", "1,1]"), ~ as.integer(endsWith(names_sigmas, .x))) %>% 
    reduce(rbind) %>% colSums()
  names_sigmas_keep <- names_sigmas[as.logical(abs(sigmas_keep - 1))]
  pars_interest_no_theta <- c("mu", names_sigmas_keep)
  pars_interest_no_theta_idx <- str_replace(pars_interest_no_theta, "\\[", "\\\\[")
  

  names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
  tbl_participant_agg$pred_theta <- colMeans(tbl_draws[, names_thetas])
  tbl_participant_agg$pred_difference <- tbl_participant_agg$pred_theta - tbl_participant_agg$prop_responses
  
  
  tbl_summary <- fit_multi$summary(variables = pars_interest)
  tbl_summary_nok <- tbl_summary %>% filter(rhat > 1.025 | rhat < 0.975)
  if (nrow(tbl_summary_nok) > 0) {
    stop(str_c(
      "participant = ", participant_sample, "; Rhat for some parameters not ok",
      "model can be found under: ", file_loc
    ))
  }
  
  file_loc <- str_c("data/infpro_task-cat_beh/models/multi-model-summary-", participant_sample, ".RDS")
  saveRDS(tbl_summary, file_loc)
  
  idx_no_theta <- map(pars_interest_no_theta_idx, ~ str_detect(tbl_summary$variable, .x)) %>%
    reduce(rbind) %>% colSums()
  tbl_label <- tbl_summary[as.logical(idx_no_theta), ]
  
  tbl_posterior <- tbl_draws %>% 
    dplyr::select(starts_with(pars_interest_no_theta), .chain) %>%
    rename(chain = .chain) %>%
    pivot_longer(starts_with(pars_interest_no_theta), names_to = "parameter", values_to = "value") %>%
    filter(parameter != "chain")
  tbl_posterior$parameter <- fct_inorder(tbl_posterior$parameter)
  
  
  pl_thetas <- plot_item_thetas(tbl_participant_agg, str_c("MV Gaussian; Participant = ", participant_sample))
  pl_posteriors <- plot_posteriors(tbl_posterior, tbl_label, n_cols = 6)
  pl_pred_uncertainty <- plot_proportion_responses(tbl_participant_agg, participant_sample, color_pred_difference = TRUE)
  

  
  # save plots
  c_names <- function(x, y) str_c("data/infpro_task-cat_beh/model-plots/", x, y, ".png")
  l_pl_names <- map(
    c("multi-gaussian-thetas-", "multi-gaussian-posteriors-", "multi-gaussian-uncertainty-"),
    c_names, y = participant_sample
  )
  l_pl <- list(pl_thetas, pl_posteriors, pl_pred_uncertainty)
  l_vals_size <- list(c(3.5, 3.5), c(11, 5.5), c(7.5, 7.5))
  pwalk(list(l_pl, l_pl_names, l_vals_size), save_my_png)
  
  return(loo_multi)
}
