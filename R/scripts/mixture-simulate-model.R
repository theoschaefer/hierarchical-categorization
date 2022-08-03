library(tidyverse)
library(mvtnorm)
library(cmdstanr)
library(rutils)


# idea: simulate observations from given model and model parameters
# try to recover the data-generating model using stan


# 1D Case -----------------------------------------------------------------


# two uni variate normal distributions
# probability of choosing right category depends on ratio of probability densities

n_trials <- 4
vals <- seq(-10, 10, by = .1)
tbl_params <- tibble(
  mean = c(2, 5),
  sd = c(1, 1)
)

my_dnorm <- function(val, mean, sd) {
  map_dbl(val, dnorm, mean = mean, sd = sd)
}




tbl_cat <- pmap(tbl_params, my_dnorm, val = vals) %>%
  reduce(cbind) %>% as_tibble()
colnames(tbl_cat) <- c("p1", "p2")
tbl_cat$x <- as.character(round(vals, 1))
tbl_cat$pcat1 <- tbl_cat$p1 / (tbl_cat$p1 + tbl_cat$p2)
tbl_cat$pcat2 <- 1 - tbl_cat$pcat1

my_samples <- function(n, tbl_df) {
  samples_tbl <- function(n, mean, sd) {
    s <- seq(mean - 2*sd, mean + 2*sd, by = .1)
    tibble(samples = s, mean = mean, sd = sd)
  }
  pmap(tbl_df, samples_tbl, n = n)
}
# now for each category n_trials for a given x value (i.e., vals) has to be generated
tbl_cat_samples <- my_samples(n_trials, tbl_params) %>%
  reduce(rbind) %>% mutate(
    cat = factor(mean, labels = seq(1, nrow(tbl_params))),
    n_trials = n_trials,
    samples = as.character(samples),
  ) %>% left_join(
    tbl_cat,
    by = c("samples" = "x")
  ) %>% mutate(
    samples = as.numeric(samples)
  )

my_rbinom <- function(n_trials, pcat1) {
  rbinom(1, n_trials, pcat1)
}

tbl_cat_samples$n_cat1 <- pmap_dbl(tbl_cat_samples[, c("n_trials", "pcat1")], my_rbinom)
tbl_cat_samples$n_cat2 <- n_trials - tbl_cat_samples$n_cat1

my_select <- function(c1, c2, cat) {
  c(c1, c2)[cat]
}

tbl_cat_samples$n_correct <- pmap_dbl(
  tbl_cat_samples[, c("n_cat1", "n_cat2", "cat")] %>% rename(c1 = n_cat1, c2 = n_cat2),
  my_select
)
tbl_cat_samples$theta_true <- pmap_dbl(
  tbl_cat_samples[, c("pcat1", "pcat2", "cat")] %>% rename(c1 = pcat1, c2=pcat2),
  my_select
)
tbl_cat_samples$samples_z <- round(scale(tbl_cat_samples$samples)[,1], 1)

# now for every value 
ggplot(tbl_cat_samples, aes(samples_z, n_correct, group = cat)) +
  geom_col(aes(fill = cat)) + facet_wrap(~ cat)




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
 vector[K] mu; //category means
 vector<lower=0>[K] sigma; //variance
}

transformed parameters {
  vector<lower=0,upper=1>[N] theta;
  matrix[N,K] lps;
  
  for (n in 1:N){
     for (k in 1:K){
        //increment log probability of the gaussian
        lps[n, k] = normal_lpdf(y[n] | mu[k], sigma[k]); 
     }
     theta[n] = exp(lps[n,cat[n]] - log_sum_exp(lps[n,]));
     //target += exp(lps[n,cat[n]] - log_sum_exp(lps[n,]));
  }
}

model {

 for(k in 1:K){
 mu[k] ~ normal(0, 3);
 sigma[k] ~ exponential(1);
 }

 n_correct ~ binomial(n_trials, theta);
}

generated quantities {
 array[N] int n_correct_predict;

  n_correct_predict = binomial_rng(n_trials, theta);
}
")

l_data <- list(
  D = 1,
  K = 2,
  N = nrow(tbl_cat_samples),
  n_trials = tbl_cat_samples$n_trials,
  n_correct = tbl_cat_samples$n_correct,
  cat = as.numeric(tbl_cat_samples$cat),
  y = tbl_cat_samples[, c("samples_z")] %>% as.data.frame() %>% as.matrix()
)
mod_1d_cat <- cmdstan_model(stan_cat_1d)
fit <- mod_1d_cat$sample(
  data = l_data, iter_sampling = 5000, iter_warmup = 5000, chains = 1
)
tbl_draws <- fit$draws(variables = c("mu", "sigma", "theta"), format = "df")
idx_params <- map(c("mu", "sigma"), ~ str_detect(names(tbl_draws), .x)) %>%
  reduce(rbind) %>% colSums()
names_params <- names(tbl_draws)[as.logical(idx_params)]
tbl_posterior <- tbl_draws[, c(all_of(names_params), ".chain")] %>% 
  rename(chain = .chain) %>%
  pivot_longer(all_of(names_params), names_to = "parameter", values_to = "value")
kd <- rutils::estimate_kd(tbl_posterior, names_params)
l <- sd_bfs(tbl_posterior, names_params, sqrt(2)/4)
map(names_params, plot_posterior, tbl = tbl_posterior, tbl_thx = l[[2]])


tbl_summary <- fit$summary(variables = c("mu", "sigma", "theta"))
names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
tbl_cat_samples$pred_theta <- colMeans(tbl_draws[, names_thetas])


ggplot(tbl_cat_samples, aes(pred_theta, theta_true)) +
  geom_point() +
  theme_bw() +
  labs(x = "Predicted Theta", y = "True Theta")



# 2D Case -----------------------------------------------------------------

# two bi-variate normal distributions
# probability of choosing right category depends on ratio of probability densities

n_trials <- 4
vals <- seq(-10, 10, by = .25)
grid_vals <- crossing(x1 = vals, x2 = vals)
grid_vals$val <- matrix(c(grid_vals$x1, grid_vals$x2), ncol = 2, byrow = FALSE)

tbl_params <- tibble(
  v_mean = c(list(c(-1.25, -1.25)), list(c(1.25, 1.25))),
  m_vcov = c(list(matrix(c(1, .3, .3, 1), nrow = 2)), list(matrix(c(1, .3, .3, 1), nrow = 2))),
  cat = factor(c(1, 2))
)

my_mvdnorm <- function(val, v_mean, m_vcov) {
  apply(val, 1, dmvnorm, mean = v_mean, sigma = m_vcov)
}



tbl_cat <- pmap(tbl_params[, c("v_mean", "m_vcov")], my_mvdnorm, val = grid_vals$val) %>%
  reduce(cbind) %>% as_tibble()
colnames(tbl_cat) <- c("p1", "p2")
tbl_x <- as.data.frame(round(grid_vals$val, 1))
names(tbl_x) <- c("x1", "x2")
tbl_cat <- as_tibble(cbind(tbl_cat, tbl_x))
tbl_cat$pcat1 <- tbl_cat$p1 / (tbl_cat$p1 + tbl_cat$p2)
tbl_cat$pcat2 <- 1 - tbl_cat$pcat1
ggplot(tbl_cat, aes(x1, x2)) + 
  geom_contour(aes(z = p1)) +
  geom_contour(aes(z = p2)) +
  # geom_tile(aes(fill = pcat1)) +
  # scale_fill_viridis_c() +
  theme_bw()
tbl_cat$x1 <- as.character(tbl_cat$x1)
tbl_cat$x2 <- as.character(tbl_cat$x2)



my_samples <- function(n, tbl_df) {
  samples_tbl <- function(n, mean, sd, cat) {
    s <- seq(mean[1] - 2*sd[1,1], mean[1] + 2*sd[1,1], by = .25)
    s2 <- crossing(x1 = s, x2 = s)
    tibble(s2, v_mean = list(mean), m_vcov = list(sd), cat = cat)
  }
  pmap(tbl_df, samples_tbl, n = n)
}

tbl_cat_samples <- my_samples(
  n_trials, tbl_params %>% rename(mean = v_mean, sd = m_vcov)
  )  %>%
  reduce(rbind) %>% mutate(
    n_trials = n_trials,
    x1 = as.character(round(x1, 1)),
    x2 = as.character(round(x2, 1))
  ) %>% left_join(
    tbl_cat,
    by = c("x1", "x2")
  ) %>% mutate(
    x1 = as.numeric(x1),
    x2 = as.numeric(x2)
  )


tbl_cat_samples$n_cat1 <- pmap_dbl(tbl_cat_samples[, c("n_trials", "pcat1")], my_rbinom)
tbl_cat_samples$n_cat2 <- n_trials - tbl_cat_samples$n_cat1

tbl_cat_samples$n_correct <- pmap_dbl(
  tbl_cat_samples[, c("n_cat1", "n_cat2", "cat")] %>% rename(c1 = n_cat1, c2 = n_cat2),
  my_select
)
tbl_cat_samples$theta_true <- pmap_dbl(
  tbl_cat_samples[, c("pcat1", "pcat2", "cat")] %>% rename(c1 = pcat1, c2=pcat2),
  my_select
)
tbl_cat_samples$x1_z <- round(scale(tbl_cat_samples$x1)[,1], 1)
tbl_cat_samples$x2_z <- round(scale(tbl_cat_samples$x2)[,1], 1)

# now for every value 
ggplot(tbl_cat_samples, aes(x1_z, x2_z, group = cat)) +
  geom_contour_filled(aes(z = theta_true, color = theta_true)) +
  facet_wrap(~ cat) + theme_bw()

