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
  real <lower=0,upper=1> w;
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
      s[i, j] = exp(-square(c)*(w*square(d1[i, j])+(1-w)*square(d2[i, j])));
      sumsim[i, cat[j]] = sumsim[i, cat[j]] + s[i,j] * bs[cat[j]];
    }
    theta[i] = sumsim[i, cat[i]] / sum(sumsim[i, ]);
  }
}

model {
  n_correct ~ binomial(n_trials, theta);
  c ~ uniform(0, 10);
  w ~ beta(1, 1);

}

generated quantities {
  array[n_stim] real log_lik_pred;

  for (n in 1:n_stim) {
  log_lik_pred[n] = binomial_lpmf(n_correct_predict[n] | n_trials_per_item[n], theta[n]);
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




aggregate_by_stimulus_and_response <- function(tbl_stim_id, tbl_train) {
  #' aggregate responses by participant, stimulus id, category, and response
  #' 
  #' @description make sure categories not responded to are filled with 0s
  #' 
  #' @param tbl_stim_id tbl_df containing all training stim_id with 
  #' associated x values and categories
  #' @param tbl_train tbl df with all category learning training data
  #' @return aggregated tbl df
  #' 
  
  tbl_design <- tbl_stim_id %>% 
    crossing(
      response = unique(tbl_train$response), 
      participant = unique(tbl_train$participant)
    ) %>%
    relocate(stim_id, .before = category)
  
  tbl_train_agg <- tbl_train_last %>% 
    group_by(participant, stim_id, d1i, d2i, d1i_z, d2i_z, category, response) %>%
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
  #' @mod_gcm the compiled cmdstanr model
  #' @return loo
  #'

  participant_sample <- tbl_participant$participant[1]
  tbl_gcm <- tbl_participant %>% 
    filter(category == response) %>% 
    mutate(prop_correct = prop_responses)
  
  # compute pairwise distances
  m_distances_x1 <- map(1:nrow(tbl_gcm), distance_1d, tbl_gcm, "d1i_z") %>% unlist() %>%
    matrix(byrow = TRUE, nrow = nrow(tbl_gcm), ncol = nrow(tbl_gcm))
  m_distances_x2 <- map(1:nrow(tbl_gcm), distance_1d, tbl_gcm, "d2i_z") %>% unlist() %>%
    matrix(byrow = TRUE, nrow = nrow(tbl_gcm), ncol = nrow(tbl_gcm))
  
  l_data <- list(
    n_stim = nrow(tbl_gcm), n_trials = tbl_gcm$n_trials, 
    n_correct = tbl_gcm$n_responses, n_cat = length(unique(tbl_gcm$category)),
    cat = as.numeric(factor(tbl_gcm$category, labels = c(1, 2, 3))),
    d1 = m_distances_x1, d2 = m_distances_x2,
    n_correct_predict = tbl_gcm$n_responses,
    n_trials_per_item = tbl_gcm$n_trials
  )
  
  fit_gcm <- mod_gcm$sample(
    data = l_data, chains = l_stan_params$n_chains, 
    iter_sampling = l_stan_params$n_samples, iter_warmup = l_stan_params$n_warmup
  )
  
  file_loc <- str_c("data/infpro_task-cat_beh/gcm-model-", participant_sample, ".RDS")
  fit_gcm$save_object(file = file_loc)
  
  pars_interest <- c("theta", "bs", "c", "w")
  pars_interest_no_theta <- c("bs", "c", "w")
  tbl_draws <- fit_gcm$draws(variables = pars_interest, format = "df")
  
  names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
  tbl_gcm$pred_theta <- colMeans(tbl_draws[, names_thetas])
  tbl_gcm$pred_difference <- tbl_gcm$pred_theta - tbl_gcm$prop_responses
  
  
  tbl_summary <- fit_gcm$summary(variables = pars_interest)
  tbl_summary_nok <- tbl_summary %>% filter(rhat > 1.1 | rhat < 0.9)
  if (nrow(tbl_summary_nok) > 0) {
    stop(str_c(
      "participant = ", participant_sample, "; Rhat for some parameters not ok",
      "model can be found under: ", file_loc
    ))
  }
  
  loo_gcm <- fit_gcm$loo(variables = "log_lik_pred")
  
  idx_no_theta <- map(pars_interest_no_theta, ~ str_detect(tbl_summary$variable, .x)) %>%
    reduce(rbind) %>% colSums()
  tbl_label <- tbl_summary[as.logical(idx_no_theta), ]
  
  tbl_posterior <- tbl_draws %>% 
    select(starts_with(pars_interest_no_theta), .chain) %>%
    rename(chain = .chain) %>%
    pivot_longer(starts_with(pars_interest_no_theta), names_to = "parameter", values_to = "value") %>%
    filter(parameter != "chain")
  tbl_posterior$parameter <- fct_inorder(tbl_posterior$parameter)
  
  pl_thetas <- plot_item_thetas(tbl_gcm, str_c("GCM; Participant = ", participant_sample))
  pl_posteriors <- plot_posteriors(tbl_posterior)
  pl_pred_uncertainty <- plot_proportion_responses(tbl_gcm, color_pred_difference = TRUE)
  
  # save plots
  c_names <- function(x, y) str_c("data/infpro_task-cat_beh/model-plots/", x, y, ".png")
  l_pl_names <- map(c("gcm-thetas-", "gcm-posteriors-", "gcm-uncertainty-"), c_names, y = participant_sample)
  l_pl <- list(pl_thetas, pl_posteriors, pl_pred_uncertainty)
  l_vals_size <- list(c(3, 3), c(5, 5), c(5.5, 5.5))
  pwalk(list(l_pl, l_pl_names, l_vals_size), save_my_png)
  
  return(loo_gcm)
}
