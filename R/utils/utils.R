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
  stan_cat_2d <- write_stan_file("
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

 for(k in 1:K){
   mu[k] ~ normal(0, 3);
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
