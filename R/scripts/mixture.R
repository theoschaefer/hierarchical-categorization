library(tidyverse)
library(cmdstanr)
library(rutils)

dir_home_grown <- c("R/utils/ellipse-utils.R", "R/utils/utils.R")
walk(dir_home_grown, source)
check_cmdstan_toolchain()



# 1D Example --------------------------------------------------------------


# generate some one-dimensional data from mixture distribution

n_data <- 400
p_mixture <- c(.8, .2)
n_mixture <- n_data * p_mixture
tbl_parms <- tibble(
  n = n_mixture,
  mean = c(0, 6),
  sd = c(2, 1),
  c = 1:length(n)
)

my_rnorm <- function(n, mean, sd, c) {
  s <- rnorm(n, mean, sd)
  tibble(component = factor(c), samples = s)
}

tbl_samples <- pmap(tbl_parms, my_rnorm) %>%
  reduce(rbind)

ggplot(tbl_samples, aes(samples, group = component)) +
  geom_density(aes(color = component)) +
  scale_color_brewer(palette = "Set1") +
  theme_bw()

# https://mc-stan.org/docs/stan-users-guide/summing-out-the-responsibility-parameter.html

stan_mixture_1d <- write_stan_file("
data {
  int<lower=1> K;          // number of mixture components
  int<lower=1> N;          // number of data points
  array[N] real y;         // observations
}
parameters {
  simplex[K] theta;          // mixing proportions
  ordered[K] mu;             // locations of mixture components
  vector<lower=0>[K] sigma;  // scales of mixture components
}
model {
  vector[K] log_theta = log(theta);  // cache log calculation
  sigma ~ lognormal(0, 2);
  mu ~ normal(0, 10);
  for (n in 1:N) {
    vector[K] lps = log_theta;
    for (k in 1:K) {
      lps[k] += normal_lpdf(y[n] | mu[k], sigma[k]);
    }
    target += log_sum_exp(lps);
  }
}
")
mod <- cmdstan_model(stan_mixture_1d)
vars <- mod$variables()
names(vars$data)
mod$exe_file()

l_data <- list(
  K = 2,
  N = nrow(tbl_samples),
  y = tbl_samples$samples
)

fit <- mod$sample(data = l_data, iter_sampling = 500, iter_warmup = 500)
tbl_draws <- fit$draws(variables = c("mu", "sigma", "theta"), format = "df")
idx_params <- map(c("mu", "sigma", "theta"), ~ str_detect(names(tbl_draws), .x)) %>%
  reduce(rbind) %>% colSums()
names_params <- names(tbl_draws)[as.logical(idx_params)]
tbl_posterior <- tbl_draws[, c(names_params, ".chain")] %>% 
  rename(chain = .chain) %>%
  pivot_longer(names_params, names_to = "parameter", values_to = "value")
kd <- rutils::estimate_kd(tbl_posterior, names_params)
l <- sd_bfs(tbl_posterior, names_params, sqrt(2)/4)
rutils::plot_posterior("mu[1]", tbl_posterior, l[[2]])
map(names_params, plot_posterior, tbl = tbl_posterior, tbl_thx = l[[2]])



# 2D Example --------------------------------------------------------------

## using true category structure to see whether model can recover parameters

# Create Categorization Data ----------------------------------------------
l_info <- list(
  n_stimuli = 100, n_categories = 2, category_shape = "ellipses", 
  cat_type = "exemplar", is_reward = FALSE
)
l_tmp <- make_stimuli(l_info)
tbl <- l_tmp[[1]]
l_info <- l_tmp[[2]]
n_trials_per_stim <- 4
tbl <- tbl %>% mutate(
  x1 = scale(x1)[, 1],
  x2 = scale(x2)[, 1],
  n_correct = 3,#ceiling(exp(-(abs(x1) + abs(x2))) * n_trials_per_stim),
  n_trials = n_trials_per_stim
)
#tbl$n_correct[tbl$category == 1] <- abs(tbl$n_correct[tbl$category == 1] - max(tbl$n_correct))
ggplot(tbl, aes(x1, x2, group = category)) +
  geom_raster(aes(fill = category, alpha = n_correct)) +
  guides(fill = "none") +
  theme_bw()


# https://maggielieu.com/2017/03/21/multivariate-gaussian-mixture-model-done-properly/
stan_mixture_2d <- write_stan_file("
data {
 int D; //number of dimensions
 int K; //number of gaussians
 int N; //number of data
 vector[D] y[N]; //data
}

parameters {
 simplex[K] theta; //mixing proportions
 ordered[D] mu[K]; //mixture component means
 cholesky_factor_corr[D] L[K]; //cholesky factor of covariance
}

model {
 real ps[K];
 
 for(k in 1:K){
 mu[k] ~ normal(0,3);
 L[k] ~ lkj_corr_cholesky(D);
 }
 

 for (n in 1:N){
 for (k in 1:K){
 ps[k] = log(theta[k])+multi_normal_cholesky_lpdf(y[n] | mu[k], L[k]); //increment log probability of the gaussian
 }
 target += log_sum_exp(ps);
 }
}

generated quantities {
 corr_matrix[D] Sigma[K];
 for (k in 1:K){
 Sigma[k] = multiply_lower_tri_self_transpose(L[k]);
 }
 
}
")


tbl_cat2 <- filter(tbl, category == "2")
tbl_cat2shift <- tbl_cat2 %>%
  mutate(
    x1 = x1 + 4,
    x2 = x2 + 4,
    category = "3")
tbl_two_ellipses <- rbind(tbl_cat2, tbl_cat2shift)

ggplot(tbl_two_ellipses, aes(x1, x2, group = category)) +
  geom_raster(aes(fill = category, alpha = n_correct)) +
  guides(fill = "none") +
  theme_bw()
l_data <- list(
  D = 2,
  K = 2,
  N = nrow(tbl_two_ellipses),
  n_trials = tbl_two_ellipses$n_trials,
  y = tbl_two_ellipses[, c("x1", "x2")]
)
mod_2d <- cmdstan_model(stan_mixture_2d)
fit <- mod_2d$sample(data = l_data, iter_sampling = 5000, iter_warmup = 2000, chains = 1)
tbl_draws <- fit$draws(variables = c("mu", "Sigma", "theta"), format = "df")
idx_params <- map(c("mu", "Sigma", "theta"), ~ str_detect(names(tbl_draws), .x)) %>%
  reduce(rbind) %>% colSums()
names_params <- names(tbl_draws)[as.logical(idx_params)]
tbl_posterior <- tbl_draws[, c(all_of(names_params), ".chain")] %>% 
  rename(chain = .chain) %>%
  pivot_longer(names_params, names_to = "parameter", values_to = "value")
kd <- rutils::estimate_kd(tbl_posterior, names_params)
l <- sd_bfs(tbl_posterior, names_params, sqrt(2)/4)
rutils::plot_posterior("L[1,2,1]", tbl_posterior, l[[2]]) + coord_cartesian(xlim = c(0, 1))
map(names_params, plot_posterior, tbl = tbl_posterior, tbl_thx = l[[2]])


fit$summary(variables = c("mu", "Sigma"))



# Actual Model ------------------------------------------------------------


stan_mixture_cat_2d <- write_stan_file("
data {
 int D; //number of dimensions
 int K; //number of gaussians
 int N; //number of data
 array[N] int n_trials; // number of trials per item
 array[N] int n_correct; // number of true categorization responses per item
 array[N] int cat; // true category labels
 matrix[D, N] y; //data
}

parameters {
 array[K] ordered[D] mu; //mixture component means
 array[K] cholesky_factor_corr[D] L; //cholesky factor of covariance
}

transformed parameters {
  vector<lower=0,upper=1>[N] theta;
  matrix[N,K] ps;
  
  for (n in 1:N){
     for (k in 1:K){
        //increment log probability of the gaussian
        ps[n, k] = multi_normal_cholesky_lpdf(y[n] | mu[k], L[k]); 
     }
     theta[n] = exp(ps[n,cat[n]] - log_sum_exp(ps[n,]));
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
 for (k in 1:K){
 Sigma[k] = multiply_lower_tri_self_transpose(L[k]);
 }
 
}
")

l_data <- list(
  D = 2,
  K = 2,
  N = nrow(tbl_two_ellipses),
  n_trials = tbl_two_ellipses$n_trials,
  n_correct = tbl_two_ellipses$n_correct,
  cat = as.numeric(tbl_two_ellipses$category) - 1,
  y = tbl_two_ellipses[, c("x1", "x2")]
)
mod_2d_cat <- cmdstan_model(stan_mixture_cat_2d)
fit <- mod_2d_cat$sample(data = l_data, iter_sampling = 5000, iter_warmup = 2000, chains = 1)
tbl_draws <- fit$draws(variables = c("mu", "Sigma", "theta"), format = "df")
idx_params <- map(c("mu", "Sigma", "theta"), ~ str_detect(names(tbl_draws), .x)) %>%
  reduce(rbind) %>% colSums()
names_params <- names(tbl_draws)[as.logical(idx_params)]
tbl_posterior <- tbl_draws[, c(all_of(names_params), ".chain")] %>% 
  rename(chain = .chain) %>%
  pivot_longer(names_params, names_to = "parameter", values_to = "value")
kd <- rutils::estimate_kd(tbl_posterior, names_params)
l <- sd_bfs(tbl_posterior, names_params, sqrt(2)/4)
rutils::plot_posterior("L[1,2,1]", tbl_posterior, l[[2]]) + coord_cartesian(xlim = c(0, 1))
map(names_params, plot_posterior, tbl = tbl_posterior, tbl_thx = l[[2]])


fit$summary(variables = c("mu", "Sigma"))
