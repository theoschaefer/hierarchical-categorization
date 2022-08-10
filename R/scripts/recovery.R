library(cmdstanr)
library(rutils)
library(tidyverse)

dir_home_grown <- c("R/utils/ellipse-utils.R", "R/utils/utils.R")
walk(dir_home_grown, source)

check_cmdstan_toolchain(fix = TRUE, quiet = TRUE)
check_cmdstan_toolchain()


# Naive Bayes -------------------------------------------------------------


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
   theta[n] = (exp(LL1_unique[cat_true[n]] - log_sum_exp(LL1_unique)) + 
   exp(LL2_unique[cat_true[n]] - log_sum_exp(LL2_unique))) / 2;
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
  //target += (LL1[cat[n]] - log_sum_exp(LL1)) + (LL2[cat[n]] - log_sum_exp(LL2));
  target += LL1[cat[n]] + LL2[cat[n]];
 }
}

")
}


library(MASS)

m_identity <- matrix(c(1, 0, 0, 1), nrow = 2)
tbl_cluster_params <- tibble(
  mu = list(c(-3, -3), c(0, 0), c(3, 3)),
  Sigma = list(m_identity, m_identity, m_identity)
)
my_mvrnorm <- function(mu, Sigma, n) {
  tbl <- as_tibble(mvrnorm(n, mu, Sigma))
  colnames(tbl) <- c("x1", "x2")
  tbl$cond <- mu[1]
  return(tbl)
}

tbl_cluster <- pmap(tbl_cluster_params, my_mvrnorm, n = 100) %>%
  reduce(rbind) %>%
  mutate(
    x1_z = scale(x1)[, 1], x2_z = scale(x2)[, 1],
    category = fct_inseq(factor(cond)),
    category = factor(category, labels = c(1, 2, 3))
  )

tbl_cluster_new <- pmap(tbl_cluster_params, my_mvrnorm, n = 10) %>%
  reduce(rbind) %>%
  mutate(
    x1_z = scale(x1)[, 1], x2_z = scale(x2)[, 1],
    category = fct_inseq(factor(cond)),
    category = factor(category, labels = c(1, 2, 3))
  )

tbl_cluster %>% ggplot(aes(x1_z, x2_z, group = as.factor(cond))) + 
  geom_point(aes(color = as.factor(cond)), show.legend = FALSE) +
  theme_bw()

l_data <- list(
  D = 2, K = length(unique(tbl_cluster$category)),
  N = nrow(tbl_cluster),
  y = tbl_cluster[, c("x1_z", "x2_z")] %>% as.matrix(),
  cat = as.numeric(tbl_cluster$category),
  cat_true = as.numeric(tbl_cluster_new$category),
  n_stim = nrow(tbl_cluster_new),
  y_unique = tbl_cluster_new[, c("x1_z", "x2_z")] %>% as.matrix()
)

stan_naive <- write_gaussian_naive_bayes_stan()
mod_naive <- cmdstan_model(stan_naive)

fit_naive <- mod_naive$sample(
  data = l_data, iter_sampling = 2000, iter_warmup = 2000, chains = 1
)

file_loc <- str_c("data/infpro_task-cat_beh/gcm-model-", participant_sample, ".RDS")
fit_naive$save_object(file = file_loc)
pars_interest <- c("mu", "sigma", "theta")
tbl_draws <- fit_naive$draws(variables = pars_interest, format = "df")
tbl_summary <- fit_naive$summary(variables = pars_interest)
names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
tbl_cluster_new$pred_theta <- colMeans(tbl_draws[, names_thetas])

plot_item_thetas(tbl_cluster_new %>% mutate(prop_correct = 1), "Gaussian")

tbl_posterior <- tbl_draws %>% 
  dplyr::select(starts_with(c("mu", "sigma")), .chain) %>%
  rename(chain = .chain) %>%
  pivot_longer(starts_with(c("mu", "sigma")), names_to = "parameter", values_to = "value")

ggplot(tbl_posterior, aes(value)) +
  geom_density(aes(color = parameter)) +
  facet_wrap(~ parameter, scales = "free_y")


tbl_cluster_new %>% grouped_agg(category, c(x1_z, x2_z)) %>%
  mutate(
    sd_x1_z = se_x1_z * sqrt(n),
    sd_x2_z = se_x2_z * sqrt(n)
  )

tbl_cluster_new %>% ggplot(aes(x1_z, x2_z, group = as.factor(cond))) + 
  geom_point(aes(color = as.factor(cond), alpha = pred_theta), show.legend = FALSE) +
  theme_bw() +
  labs(
    caption = "Alpha reflects prediction uncertainty"
  )


