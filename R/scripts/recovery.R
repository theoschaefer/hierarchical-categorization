library(cmdstanr)
library(rutils)
library(tidyverse)

dir_home_grown <- c("R/utils/ellipse-utils.R", "R/utils/utils.R")
walk(dir_home_grown, source)

check_cmdstan_toolchain(fix = TRUE, quiet = TRUE)
check_cmdstan_toolchain()


# Naive Bayes -------------------------------------------------------------


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

tbl_cluster <- pmap(tbl_cluster_params, my_mvrnorm, n = 30) %>%
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

stan_naive <- write_gaussian_naive_bayes_stan_recovery()
mod_naive <- cmdstan_model(stan_naive)

fit_naive <- mod_naive$sample(
  data = l_data, iter_sampling = 2000, iter_warmup = 2000, chains = 1
)

file_loc <- str_c("data/infpro_task-cat_beh/recovery-naive-model-", participant_sample, ".RDS")
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



# GCM ---------------------------------------------------------------------


# use same data as for gaussian naive bayes recovery study
l_params_model <- list(
  c = .5,
  w = .5,
  b = .5
)

l_params_simulation <- list(
  n_trials = 10
)

l_dist_sim <- gcm_distances_similarities(tbl_cluster, l_params_model)
tbl_cluster$prop_correct_true <- map_dbl(
  1:nrow(tbl_cluster), 
  gcm_response_proportions, 
  tbl_df = tbl_cluster, 
  m_sims = l_dist_sim[["m_similarities"]]
)
tbl_cluster$n_trials <- l_params_simulation[["n_trials"]]
tbl_cluster$n_true <- pmap_int(
  tbl_cluster[, c("n_trials", "prop_correct_true")], my_rbinom
)


l_data_gcm <- list(
  n_stim = nrow(tbl_cluster),
  n_cat = length(unique(tbl_cluster$category)),
  n_trials = tbl_cluster$n_trials,
  n_correct = tbl_cluster$n_true,
  cat = tbl_cluster$category %>% as.numeric(),
  d1 = l_dist_sim[["m_distances_x1"]],
  d2 = l_dist_sim[["m_distances_x2"]]
)

stan_gcm <- write_gcm_stan_file()
mod_gcm <- cmdstan_model(stan_gcm)

fit_gcm <- mod_gcm$sample(
  data = l_data_gcm, iter_sampling = 2000, iter_warmup = 2000, chains = 1
)

file_loc <- str_c("data/recovery/recovery-gcm-model.RDS")
fit_gcm$save_object(file = file_loc)
pars_interest <- c("c", "w", "b", "theta")
tbl_draws <- fit_gcm$draws(variables = pars_interest, format = "df")
tbl_summary <- fit_gcm$summary(variables = pars_interest)
names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
tbl_cluster$pred_theta <- colMeans(tbl_draws[, names_thetas])

plot_item_thetas(tbl_cluster %>% mutate(prop_correct = prop_correct_true), "GCM")

tbl_posterior <- tbl_draws %>% 
  dplyr::select(starts_with(pars_interest[pars_interest != "theta"]), .chain) %>%
  rename(chain = .chain) %>%
  pivot_longer(
    cols = pars_interest[pars_interest != "theta"], 
    names_to = "parameter", values_to = "value"
  )

ggplot(tbl_posterior, aes(value)) +
  geom_density(aes(color = parameter), show.legend = FALSE) +
  facet_wrap(~ parameter, scales = "free_y") +
  theme_bw() +
  labs(x = "Parameter Value", y = "Posterior Density")


tbl_cluster %>% grouped_agg(category, c(x1_z, x2_z)) %>%
  mutate(
    sd_x1_z = se_x1_z * sqrt(n),
    sd_x2_z = se_x2_z * sqrt(n)
  )

tbl_cluster %>% ggplot(aes(x1_z, x2_z, group = as.factor(cond))) + 
  geom_point(aes(color = as.factor(cond), alpha = pred_theta), show.legend = FALSE) +
  theme_bw() +
  labs(
    caption = "Alpha reflects prediction uncertainty"
  )

