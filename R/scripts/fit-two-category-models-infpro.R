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

# define how many trials starting from the last trial should be analyzed
tbl_train %>% count(participant) %>% arrange(n)
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


tbl_train_agg <- tbl_train %>% 
  group_by(participant, d1i, d2i, category) %>%
  summarize(
    n_trials = n(), 
    n_correct = sum(accuracy),
    prop_correct = n_correct / n_trials
  ) %>% ungroup()

participant_sample <- sample(unique(tbl_train_agg$participant), 1)
tbl_sample <- tbl_train_agg %>% filter(participant == participant_sample)
tbl_sample <- tbl_sample %>% mutate(
  d1i_z = scale(d1i)[, 1],
  d2i_z = scale(d2i)[, 1]
)


ggplot(tbl_sample, aes(d1i_z, d2i_z)) +
  geom_point(aes(size = prop_correct, color = category), show.legend = FALSE) +
  geom_label_repel(aes(label = round(prop_correct, 2)), size = 2.5) +
  theme_bw() +
  labs(x = expr(x[1]), y = expr(x[2]))


# GCM ---------------------------------------------------------------------

tbl_sample_gcm <- tbl_sample

# compute pairwise distances
m_distances_x1 <- map(1:nrow(tbl_sample_gcm), distance_1d, tbl_sample_gcm, "d1i_z") %>% unlist() %>%
  matrix(byrow = TRUE, nrow = nrow(tbl_sample_gcm), ncol = nrow(tbl_sample_gcm))
m_distances_x2 <- map(1:nrow(tbl_sample_gcm), distance_1d, tbl_sample_gcm, "d2i_z") %>% unlist() %>%
  matrix(byrow = TRUE, nrow = nrow(tbl_sample_gcm), ncol = nrow(tbl_sample_gcm))


stan_gcm <- write_gcm_stan_file()
mod_gcm <- cmdstan_model(stan_gcm)

l_data <- list(
  n_stim = nrow(tbl_sample_gcm), n_trials = tbl_sample_gcm$n_trials, 
  n_correct = tbl_sample_gcm$n_correct, n_cat = length(unique(tbl_sample_gcm$category)),
  cat = as.numeric(factor(tbl_sample_gcm$category, labels = c(1, 2, 3))),
  d1 = m_distances_x1, d2 = m_distances_x2
)


fit_gcm <- mod_gcm$sample(
  data = l_data, iter_sampling = 1000, iter_warmup = 1000, chains = 1
)
file_loc <- str_c("data/infpro_task-cat_beh/gcm-model-", participant_sample, ".RDS")
fit_gcm$save_object(file = file_loc)
pars_interest <- c("theta", "b", "c", "w")
tbl_draws <- fit_gcm$draws(variables = pars_interest, format = "df")
tbl_summary <- fit_gcm$summary(variables = pars_interest)
names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
tbl_sample_gcm$pred_theta <- colMeans(tbl_draws[, names_thetas])

plot_item_thetas(tbl_sample_gcm, "GCM")



# Bivariate Gaussian Classification Model ---------------------------------
# aka prototype model

tbl_sample_gaussian <- tbl_sample

stan_gaussian <- write_bivariate_gaussian_stan()
mod_gaussian <- cmdstan_model(stan_gaussian)


l_data <- list(
  D = 2,
  K = 3,
  N = nrow(tbl_sample_gaussian),
  n_trials = tbl_sample_gaussian$n_trials,
  n_correct = tbl_sample_gaussian$n_correct,
  cat = as.numeric(factor(tbl_sample_gaussian$category, labels = c(1, 2, 3))),
  y = tbl_sample_gaussian[, c("d1i_z", "d2i_z")] %>% as.data.frame() %>% as.matrix()
)
fit_gaussian <- mod_gaussian$sample(
  data = l_data, iter_sampling = 3000, iter_warmup = 3000, chains = 1
)
file_loc <- str_c(
  "data/infpro_task-cat_beh/gaussian-model-", participant_sample, ".RDS"
)
fit_gaussian$save_object(file = file_loc)
tbl_draws <- fit_gaussian$draws(variables = c("mu", "Sigma", "theta"), format = "df")
tbl_summary <- fit_gaussian$summary(variables = c("mu", "Sigma", "theta"))
names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
tbl_sample_gaussian$pred_theta <- colMeans(tbl_draws[, names_thetas])


plot_item_thetas(tbl_sample_gaussian, "Gaussian")

vars_extract <- startsWith(tbl_summary$variable, "mu") | 
  startsWith(tbl_summary$variable, "Sigma")

tbl_summary[vars_extract, ] %>%
  arrange(variable)
