library(tidyverse)
library(cmdstanr)
library(rutils)
library(ggrepel)
library(gridExtra)

utils_loc <- c("R/utils/plotting-utils.R", "R/utils/utils.R")
walk(utils_loc, source)

file_loc_train <- "data/infpro_task-cat_beh/infpro_task-cat_beh.csv"
file_loc_transfer <- "data/infpro_task-cat_beh/infpro_task-cat2_beh.csv"
tbl_train <- read_csv(file_loc_train, show_col_types = FALSE)
tbl_transfer <- read_csv(file_loc_transfer, show_col_types = FALSE)

tbl_train <- tbl_train  %>% 
  mutate(
    d1i_z = scale(d1i)[, 1],
    d2i_z = scale(d2i)[, 1]
  )

tbl_stim_id <- tbl_train %>% count(d1i, d2i, d1i_z, d2i_z, category) %>%
  arrange(d1i, d2i) %>% mutate(stim_id = seq_along(d1i + d2i)) %>%
  dplyr::select(-n)
tbl_train <- tbl_train %>% 
  left_join(tbl_stim_id, by = c("d1i", "d2i", "d1i_z", "d2i_z", "category"))

# define how many trials starting from the last trial should be analyzed
n_last_trials <- 500

tbl_train_last <- tbl_train %>% group_by(participant) %>%
  mutate(
    rwn_fwd = row_number(block),
    rwn_bkwd = row_number(desc(rwn_fwd))
  ) %>% ungroup() %>%
  filter(rwn_bkwd <= n_last_trials)


# plot average accuracy across participants in train and transfer tests
pl_train <- plot_average_categorization_accuracy(tbl_train_last, "Train")
pl_tf <- plot_average_categorization_accuracy(tbl_transfer, "Transfer")
marrangeGrob(list(pl_train, pl_tf), ncol = 2, nrow = 1)

tbl_train_agg <- aggregate_by_stimulus_and_response(tbl_stim_id, tbl_train)


participant_sample <- sample(unique(tbl_train_agg$participant), 1)
tbl_sample <- tbl_train_agg %>% filter(participant == participant_sample)
tbl_train_last <- tbl_train_last %>% filter(participant == participant_sample)

plot_proportion_responses(
  tbl_sample %>% mutate(response = str_c("Response = ", response)) %>%
    filter(n_responses > 0),
  facet_by_response = TRUE
)

# GCM ---------------------------------------------------------------------
# aka exemplar model


tbl_sample_gcm <- tbl_sample %>% 
  filter(category == response) %>% 
  mutate(prop_correct = prop_responses)

plot_proportion_responses(tbl_sample_gcm)

# compute pairwise distances
m_distances_x1 <- map(1:nrow(tbl_sample_gcm), distance_1d, tbl_sample_gcm, "d1i_z") %>% unlist() %>%
  matrix(byrow = TRUE, nrow = nrow(tbl_sample_gcm), ncol = nrow(tbl_sample_gcm))
m_distances_x2 <- map(1:nrow(tbl_sample_gcm), distance_1d, tbl_sample_gcm, "d2i_z") %>% unlist() %>%
  matrix(byrow = TRUE, nrow = nrow(tbl_sample_gcm), ncol = nrow(tbl_sample_gcm))


stan_gcm <- write_gcm_stan_file()
mod_gcm <- cmdstan_model(stan_gcm)

l_data <- list(
  n_stim = nrow(tbl_sample_gcm), n_trials = tbl_sample_gcm$n_trials, 
  n_correct = tbl_sample_gcm$n_responses, n_cat = length(unique(tbl_sample_gcm$category)),
  cat = as.numeric(factor(tbl_sample_gcm$category, labels = c(1, 2, 3))),
  d1 = m_distances_x1, d2 = m_distances_x2
)

fit_gcm <- mod_gcm$sample(
  data = l_data, iter_sampling = 5000, iter_warmup = 5000, chains = 1
)
file_loc <- str_c("data/infpro_task-cat_beh/gcm-model-", participant_sample, ".RDS")
fit_gcm$save_object(file = file_loc)


pars_interest <- c("theta", "bs", "c", "w")
pars_interest_no_theta <- c("bs", "c", "w")
tbl_draws <- fit_gcm$draws(variables = pars_interest, format = "df")

names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
tbl_sample_gcm$pred_theta <- colMeans(tbl_draws[, names_thetas])
tbl_sample_gcm$pred_difference <- tbl_sample_gcm$pred_theta - tbl_sample_gcm$prop_responses


tbl_summary <- fit_gcm$summary(variables = pars_interest)
idx_no_theta <- map(pars_interest_no_theta, ~ str_detect(tbl_summary$variable, .x)) %>%
  reduce(rbind) %>% colSums()
tbl_label <- tbl_summary[as.logical(idx_no_theta), ]

tbl_posterior <- tbl_draws %>% 
  select(starts_with(pars_interest_no_theta), .chain) %>%
  rename(chain = .chain) %>%
  pivot_longer(starts_with(pars_interest_no_theta), names_to = "parameter", values_to = "value") %>%
  filter(parameter != "chain")
tbl_posterior$parameter <- fct_inorder(tbl_posterior$parameter)



plot_item_thetas(tbl_sample_gcm, "GCM")
plot_posteriors(tbl_posterior)
plot_proportion_responses(tbl_sample_gcm, color_pred_difference = TRUE)

tbl_sample %>% group_by(response) %>% summarize(sum(n_responses))

# Bivariate Gaussian Classification Model ---------------------------------
# aka prototype model

tbl_sample_gaussian <- tbl_train_last %>%
  mutate(response_int = factor(category, labels = c(1, 2, 3)))
tbl_sample_gaussian_unique <- tbl_sample_gcm %>% 
  dplyr::select(-c(pred_theta, -pred_difference)) %>%
  mutate(response_int = factor(category, labels = c(1, 2, 3)))
stan_gaussian <- write_gaussian_naive_bayes_stan()
mod_gaussian <- cmdstan_model(stan_gaussian)

l_data <- list(
  D = 2, K = length(unique(tbl_sample_gaussian$category)),
  N = nrow(tbl_sample_gaussian),
  y = tbl_sample_gaussian[, c("d1i_z", "d2i_z")] %>% as.matrix(),
  cat = as.numeric(tbl_sample_gaussian$response_int),
  cat_true = as.numeric(tbl_sample_gaussian_unique$response_int),
  n_stim = nrow(tbl_sample_gaussian_unique),
  y_unique = tbl_sample_gaussian_unique[, c("d1i_z", "d2i_z")] %>% as.matrix()
)

fit_gaussian <- mod_gaussian$sample(
  data = l_data, iter_sampling = 5000, iter_warmup = 5000, chains = 1
)
file_loc <- str_c(
  "data/infpro_task-cat_beh/gaussian-model-", participant_sample, ".RDS"
)
fit_gaussian$save_object(file = file_loc)
fit_gaussian <- readRDS(file_loc)

pars_interest <- c("mu1", "mu2", "sigma", "theta")
pars_interest_no_theta <- c("mu1", "mu2", "sigma")
tbl_draws <- fit_gaussian$draws(variables = pars_interest, format = "df")
tbl_draws <- tbl_draws %>% rename(`mu2[1]` = `mu2[3]`, `mu2[3]` = `mu2[1]`)

names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
tbl_sample_gaussian_unique$pred_theta <- colMeans(tbl_draws[, names_thetas])
tbl_sample_gaussian_unique$pred_difference <- tbl_sample_gaussian_unique$pred_theta - tbl_sample_gaussian_unique$prop_responses


tbl_summary <- fit_gaussian$summary(variables = pars_interest)
# rename to correct for reverse ordering in stan model
old_3 <- tbl_summary$variable == "mu2[3]"
old_1 <- tbl_summary$variable == "mu2[1]"
tbl_summary$variable[old_3] <- "mu2[1]"
tbl_summary$variable[old_1] <- "mu2[3]"

idx_no_theta <- map(pars_interest_no_theta, ~ str_detect(tbl_summary$variable, .x)) %>%
  reduce(rbind) %>% colSums()
tbl_label <- tbl_summary[as.logical(idx_no_theta), ]

tbl_posterior <- tbl_draws %>% 
  dplyr::select(starts_with(pars_interest_no_theta), .chain) %>%
  rename(chain = .chain) %>%
  pivot_longer(starts_with(pars_interest_no_theta), names_to = "parameter", values_to = "value") %>%
  filter(parameter != "chain")
tbl_posterior$parameter <- fct_inorder(tbl_posterior$parameter)



plot_item_thetas(tbl_sample_gaussian_unique, "Gaussian")
plot_posteriors(tbl_posterior)
plot_proportion_responses(tbl_sample_gaussian_unique, color_pred_difference = TRUE)

tbl_sample %>% group_by(response) %>% summarize(sum(n_responses))



# Multivariate Gaussian ---------------------------------------------------


stan_gaussian_multi <- write_gaussian_multi_bayes_stan()
mod_gaussian_multi <- cmdstan_model(stan_gaussian_multi)

# l_data is same as for naive gaussian

fit_gaussian_multi <- mod_gaussian_multi$sample(
  data = l_data, iter_sampling = 2000, iter_warmup = 2000, chains = 1
)
file_loc <- str_c(
  "data/infpro_task-cat_beh/gaussian-multi-model-", participant_sample, ".RDS"
)
fit_gaussian_multi$save_object(file = file_loc)
fit_gaussian_multi <- readRDS(file_loc)

pars_interest <- c("mu", "Sigma", "theta")
pars_interest_no_theta <- c("mu", "Sigma")
tbl_draws <- fit_gaussian_multi$draws(variables = pars_interest, format = "df")

names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
tbl_sample_gaussian_unique$pred_theta <- colMeans(tbl_draws[, names_thetas])
tbl_sample_gaussian_unique$pred_difference <- tbl_sample_gaussian_unique$pred_theta - tbl_sample_gaussian_unique$prop_responses


tbl_summary <- fit_gaussian_multi$summary(variables = pars_interest)

idx_no_theta <- map(pars_interest_no_theta, ~ str_detect(tbl_summary$variable, .x)) %>%
  reduce(rbind) %>% colSums()
tbl_label <- tbl_summary[as.logical(idx_no_theta), ]

tbl_posterior <- tbl_draws %>% 
  dplyr::select(starts_with(pars_interest_no_theta), .chain) %>%
  rename(chain = .chain) %>%
  pivot_longer(starts_with(pars_interest_no_theta), names_to = "parameter", values_to = "value") %>%
  filter(parameter != "chain")
tbl_posterior$parameter <- fct_inorder(tbl_posterior$parameter)


plot_item_thetas(tbl_sample_gaussian_unique, "Gaussian")
plot_posteriors(tbl_posterior)
plot_proportion_responses(tbl_sample_gaussian_unique, color_pred_difference = TRUE)

tbl_sample %>% group_by(response) %>% summarize(sum(n_responses))
