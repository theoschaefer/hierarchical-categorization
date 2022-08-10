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
  array[n_stim] int cat; // actual category for a given stimulus
  array[n_stim, n_stim] real<lower=0> d1;
  array[n_stim, n_stim] real<lower=0> d2;
}


parameters {
  real <lower=0> c;
  real <lower=0,upper=1> w;
  real <lower=0,upper=1> b;
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
  c ~ uniform(0, 5);
  w ~ beta(1, 1);
  b ~ beta(1, 1);
  
}

generated quantities {
  array[n_stim] int n_correct_predict;
  n_correct_predict = binomial_rng(n_trials, theta);
}

")
}

write_bivariate_gaussian_stan <- function() {
  write_stan_file("
data {
 int D; //number of dimensions
 int K; //number of gaussians
 int N; //number of data
 array[N] int n_trials; // number of trials per item
 array[N] int n_correct; // number of true categorization responses per item
 array[N] int cat; // true category labels
 matrix[N, D] y; //data
}

parameters {
 array[K] row_vector[D] mu; //category means
 array[K] cholesky_factor_corr[D] L; //variance
}

transformed parameters {
  vector<lower=0,upper=1>[N] theta;
  matrix[N,K] lps;
  
  for (n in 1:N){
     for (k in 1:K){
        //increment log probability of the gaussian
        lps[n, k] = multi_normal_cholesky_lpdf(y[n] | mu[k], L[k]); 
     }
     theta[n] = exp(lps[n,cat[n]] - log_sum_exp(lps[n,]));
     //target += exp(lps[n,cat[n]] - log_sum_exp(lps[n,]));
  }
}

model {

 mu[1,1] ~ normal(-1, .3);
 mu[1,2] ~ normal(.5, .3);
 mu[2,1] ~ normal(0, .3);
 mu[2,2] ~ normal(0, .3);
 mu[3,1] ~ normal(1, .3);
 mu[3,2] ~ normal(-.5, .3);
  
 for(k in 1:K){
   L[k] ~ lkj_corr_cholesky(D);
 }

 n_correct ~ binomial(n_trials, theta);
}

generated quantities {
 array[K] corr_matrix[D] Sigma;
 array[N] int n_correct_predict;
 
 for (k in 1:K){
 Sigma[k] = multiply_lower_tri_self_transpose(L[k]);
 }
  n_correct_predict = binomial_rng(n_trials, theta);
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



gcm_response_proportions <- function(i, tbl_df, m_sims) {
  #' compute gcm response proportions given pairwise similarities
  #' 
  #' @description computes summed similarity of within category items
  #' divided by summed similarity of all items
  #' 
  #' @param i idx of the row, for which to compute the response probability
  #' @param tbl_df tbl df with x1 and x2 features and category column
  #' @param m_sims nxn matrix with pairwise similarities
  #' @return dbl with response probability of being correct
  #' 
  is_same_category <- function(i, tbl) {
    tbl$category[i] == tbl$category
  }
  
  sum_sim_same <- sum(m_sims[i, ][is_same_category(i, tbl_df)])
  sum_sim_all <- sum(m_sims[i, ])
  sum_sim_same / sum_sim_all
}


my_rbinom <- function(n_trials, prop_correct_true) {
  rbinom(n = 1, size = n_trials, prob = prop_correct_true)
}