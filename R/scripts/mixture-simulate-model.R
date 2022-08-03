library(tidyverse)
library(cmdstanr)


# idea: simulate observations from given model and model parameters
# try to recover the data-generating model using stan
# 

# one dimensional case
# two uni variate normal distributions
# probability of choosing right category depends on ratio of probability densities

n_trials <- 4
vals <- seq(0, 10, by = .1)
tbl_params <- tibble(
  mean = c(2, 5),
  sd = c(1, 1)
)

my_dnorm <- function(val, mean, sd) {
  map_dbl(val, dnorm, mean = mean, sd = sd)
}

my_samples <- function(n, tbl_df) {
  samples_tbl <- function(n, mean, sd) {
    s <- seq(mean - 2*sd, mean + 2*sd, by = .1)
    tibble(samples = s, mean = mean, sd = sd)
  }
  pmap(tbl_df, samples_tbl, n = n)
}


tbl_cat <- pmap(tbl_params, my_dnorm, val = vals) %>%
  reduce(cbind) %>% as_tibble()
colnames(tbl_cat) <- c("p1", "p2")
tbl_cat$x <- vals
tbl_cat$pcat1 <- tbl_cat$p1 / (tbl_cat$p1 + tbl_cat$p2)
tbl_cat$pcat2 <- 1 - tbl_cat$pcat1


# now for each category n_trials for a given x value (i.e., vals) has to be generated
tbl_cat_samples <- my_samples(n_trials, tbl_params) %>%
  reduce(rbind) %>% mutate(
    cat = factor(mean, labels = seq(1, nrow(tbl_params))),
    n_trials = n_trials,
    samples = as.character(samples),
  ) %>% left_join(
    tbl_cat %>% mutate(x = as.character(x)),
    by = c("samples" = "x")
  ) %>% mutate(
    samples = as.numeric(samples)
  )

my_rbinom <- function(n_trials, pcat1) {
  rbinom(1, n_trials, pcat1)
}

tbl_cat_samples$n_cat1 <- pmap_dbl(tbl_cat_samples[, c("n_trials", "pcat1")], my_rbinom)
tbl_cat_samples$n_cat2 <- n_trials - tbl_cat_samples$n_cat1

my_select <- function(n_cat1, n_cat2, cat) {
  c(n_cat1, n_cat2)[cat]
}

tbl_cat_samples$n_correct <- pmap_dbl(tbl_cat_samples[, c("n_cat1", "n_cat2", "cat")], my_select)


# now for every value 
ggplot(tbl_cat_samples, aes(samples, n, group = cat)) +
  geom_col(aes(fill = cat))




stan_cat_1d <- write_stan_file("
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
 array[K] real mu; //mixture component means
 array[K]<lower=0> real sigma; //cholesky factor of covariance
}

transformed parameters {
  vector<lower=0,upper=1>[N] theta;
  matrix[N,K] lps;
  
  for (n in 1:N){
     for (k in 1:K){
        //increment log probability of the gaussian
        lps[n, k] = normal_lpdf(y[n] | mu[k], sigma[k]); 
     }
     //theta[n] = exp(lps[n,cat[n]] - log_sum_exp(lps[n,]));
     target += exp(lps[n,cat[n]] - log_sum_exp(lps[n,]));
  }
}

model {

 for(k in 1:K){
 mu[k] ~ normal(0,3);
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

l_data <- list(
  D = 1,
  K = 2,
  N = nrow(tbl_samples_cut),
  n_trials = tbl_samples_cut$n_trials,
  n_correct = tbl_samples_cut$n_correct,
  cat = as.numeric(tbl_samples_cut$component),
  y = tbl_samples_cut[, c("y_cut")] %>% as.data.frame() %>% as.matrix()
)
mod_1d_cat <- cmdstan_model(stan_cat_1d)
fit <- mod_2d_cat$sample(
  data = l_data, iter_sampling = 2000, iter_warmup = 500, chains = 1
)
tbl_draws <- fit$draws(variables = c("ps", "mu", "Sigma", "theta"), format = "df")
idx_params <- map(c("mu", "Sigma", "theta"), ~ str_detect(names(tbl_draws), .x)) %>%
  reduce(rbind) %>% colSums()
names_params <- names(tbl_draws)[as.logical(idx_params)]
tbl_posterior <- tbl_draws[, c(all_of(names_params), ".chain")] %>% 
  rename(chain = .chain) %>%
  pivot_longer(all_of(names_params), names_to = "parameter", values_to = "value")
kd <- rutils::estimate_kd(tbl_posterior, names_params)
l <- sd_bfs(tbl_posterior, names_params, sqrt(2)/4)
rutils::plot_posterior("L[1,2,1]", tbl_posterior, l[[2]]) + coord_cartesian(xlim = c(0, 1))
map(names_params, plot_posterior, tbl = tbl_posterior, tbl_thx = l[[2]])


fit$summary(variables = c("mu", "Sigma", "theta"))



names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
tbl$pred_theta <- colMeans(tbl_draws[, names_thetas])

