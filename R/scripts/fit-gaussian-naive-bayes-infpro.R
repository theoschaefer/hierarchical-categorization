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


tbl_train_agg <- tbl_train %>% 
  group_by(participant, d1i, d2i, category, response) %>%
  summarize(
    n_responses = n(),
    n_correct = sum(accuracy)
  ) %>% group_by(participant, d1i, d2i) %>%
  mutate(
    n_trials = sum(n_responses), 
    prop_responses = n_responses / n_trials) %>%
  ungroup()

participant_sample <- sample(unique(tbl_train_agg$participant), 1)
tbl_sample <- tbl_train_agg %>% filter(participant == participant_sample)
tbl_sample <- tbl_sample %>% mutate(
  d1i_z = scale(d1i)[, 1],
  d2i_z = scale(d2i)[, 1]
)

ggplot(tbl_sample %>% mutate(category = str_c("True Cat. = ", category)), aes(d1i_z, d2i_z)) +
  geom_point(aes(size = prop_responses, color = category), show.legend = FALSE) +
  geom_label_repel(aes(label = round(prop_responses, 2)), size = 2.5) +
  #ggtitle(str_c("Participant = ", participant_sample)) +
  facet_wrap(~ response) +
  theme_bw() +
  labs(x = expr(x[1]), y = expr(x[2]))


# Gaussian Naive Bayes ----------------------------------------------------


write_gaussian_naive_bayes_stan <- function() {
  write_stan_file("
data {
 int D; //number of dimensions
 int K; //number of categories
 int N; //number of data
 array[N, K] int n_responses; //data
 matrix[N, D] y;
}

parameters {
 array[K] row_vector[D] mu; //category means
 array[K,D] real <lower=0> sigma; //variance
}

transformed parameters {
  array[N] vector[K] theta;
  matrix[N,K] lps1;
  matrix[N,K] lps2;
  
  for (n in 1:N){
     for (k in 1:K){
        lps1[n, k] = normal_lpdf(y[n,1] | mu[k][1], sigma[k][1]); 
        lps2[n, k] = normal_lpdf(y[n,2] | mu[k][2], sigma[k][2]);
     }
     for (k in 1:K){
        theta[n][k] = (exp(lps1[n,k]) + exp(lps2[n,k])) / (sum(exp(lps1[n,])) + sum(exp(lps2[n,])));
     }
  }
}

model {
  
 for(k in 1:K){
   for (d in 1:D) {
     sigma[k][d] ~ uniform(0.1, 5);
   }
 }
 mu[1, 1] ~ normal(-.5, .5);
 mu[1, 2] ~ normal(.5, .5);
 mu[2, 1] ~ normal(0, .5);
 mu[2, 2] ~ normal(0, .5);
 mu[3, 1] ~ normal(.5, .5);
 mu[3, 2] ~ normal(-.5, .5);
 
 for (n in 1:N){
  n_responses[n] ~ multinomial(theta[n]);
 }
}
")
}

tbl_sample_naive <- tbl_sample %>% 
  select(-c(n_correct, n_trials, prop_responses)) %>%
  pivot_wider(names_from = response, values_from = n_responses, values_fill = 0)

tbl_sample_naive_1 <- tbl_sample_naive[1, ]

l_data <- list(
  D = 2, K = length(unique(tbl_sample_naive$category)),
  N = nrow(tbl_sample_naive),
  n_responses = tbl_sample_naive[, c("A", "B", "C")] %>% 
    as.matrix(),
  y = tbl_sample_naive[, c("d1i_z", "d2i_z")] %>% as.matrix()
)

stan_naive <- write_gaussian_naive_bayes_stan()
mod_naive <- cmdstan_model(stan_naive)

fit_naive <- mod_naive$sample(
  data = l_data, iter_sampling = 5000, iter_warmup = 5000, chains = 1
)

file_loc <- str_c("data/infpro_task-cat_beh/gcm-model-", participant_sample, ".RDS")
fit_naive$save_object(file = file_loc)
pars_interest <- c("theta", "mu", "sigma")
tbl_draws <- fit_naive$draws(variables = pars_interest, format = "df")
tbl_summary <- fit_naive$summary(variables = pars_interest)
names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
tbl_sample_naive$pred_theta <- colMeans(tbl_draws[, names_thetas])

plot_item_thetas(tbl_sample_naive, "GCM")

tbl_posterior <- tbl_draws %>% 
  select(starts_with("theta"), .chain) %>%
  rename(chain = .chain) %>%
  pivot_longer(starts_with("theta"), names_to = "parameter", values_to = "value")

ggplot(tbl_posterior, aes(value)) +
  geom_density(aes(color = parameter)) +
  facet_wrap(~ parameter, scales = "free_y")





# Gaussian Naive Bayes V2 -------------------------------------------------


write_gaussian_naive_bayes_stan_v2 <- function() {
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

")
}


participant_sample <- sample(unique(tbl_train_agg$participant), 1)

# the following could be replaced by tbl_transfer to predict on t2
# data while training on t1 data

tbl_train_agg <- tbl_train %>% 
  group_by(participant, d1i, d2i, category) %>%
  summarize(
    n_trials = n(),
    n_correct = sum(accuracy),
    prop_correct = n_correct / n_trials
  ) %>% group_by(participant) %>% 
  mutate(
    d1i_z = scale(d1i)[, 1],
    d2i_z = scale(d2i)[, 1]
  ) %>% ungroup()
tbl_sample <- tbl_train_agg %>% filter(participant == participant_sample)
n_stim <- nrow(tbl_sample)
cat_true <- recode(tbl_sample$category, "A" = 1, "B" = 3, "C" = 2)
y_unique <- tbl_sample[, c("d1i_z", "d2i_z")] %>% as.matrix()


# recode category if rep of true category params should be sampled
# redode response if rep of given responses should be sampled
tbl_naive2 <- tbl_train %>% filter(participant == participant_sample) %>%
  mutate(
    d1i_z = scale(d1i), d2i_z = scale(d2i),
    response_int = recode(response, "A" = 1, "B" = 3, "C" = 2)
  )



l_data <- list(
  D = 2, K = length(unique(tbl_naive2$category)),
  N = nrow(tbl_naive2),
  y = tbl_naive2[, c("d1i_z", "d2i_z")] %>% as.matrix(),
  cat = tbl_naive2$response_int,
  cat_true = cat_true,
  n_stim = n_stim,
  y_unique = y_unique
)

stan_naive_v2 <- write_gaussian_naive_bayes_stan_v2()
mod_naive_v2 <- cmdstan_model(stan_naive_v2)

fit_naive_v2 <- mod_naive_v2$sample(
  data = l_data, iter_sampling = 2000, iter_warmup = 2000, chains = 1
)

file_loc <- str_c("data/infpro_task-cat_beh/gcm-model-", participant_sample, ".RDS")
fit_naive_v2$save_object(file = file_loc)
pars_interest <- c("mu1", "mu2", "sigma", "theta")
tbl_draws <- fit_naive_v2$draws(variables = pars_interest, format = "df")
tbl_summary <- fit_naive_v2$summary(variables = pars_interest)
names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
tbl_sample$pred_theta <- colMeans(tbl_draws[, names_thetas])

plot_item_thetas(tbl_sample %>% mutate(pred_theta = pred_theta), "Gaussian")

tbl_posterior <- tbl_draws %>% 
  dplyr::select(starts_with(c("mu1", "mu2", "sigma")), .chain) %>%
  rename(chain = .chain) %>%
  pivot_longer(starts_with(c("mu1", "mu2", "sigma")), names_to = "parameter", values_to = "value")

ggplot(tbl_posterior, aes(value)) +
  geom_density(aes(color = parameter)) +
  facet_wrap(~ parameter, scales = "free_y")

tbl_sample %>% ggplot(aes(d1i_z, d2i_z, group = as.factor(category))) + 
  geom_point(aes(color = as.factor(category), alpha = pred_theta, size = prop_correct)) +
  theme_bw() +
  labs(
    caption = "Alpha reflects prediction uncertainty"
  )

n_sds <- 1
ggplot(tbl_sample %>% mutate(category = str_c("True Cat. = ", category)), aes(d1i_z, d2i_z)) +
  geom_point(aes(size = prop_correct, color = category)) +
  # ggforce::geom_ellipse(aes(x0=-.48, y0=.37, a=n_sds*(.74), b=n_sds*(.78), angle=0)) +
  # ggforce::geom_ellipse(aes(x0=.37, y0=-.13, a=n_sds*(.76), b=n_sds*(.72), angle=0)) +
  # ggforce::geom_ellipse(aes(x0=-.87, y0=1.78, a=n_sds*(1.24), b=n_sds*(1.39), angle=0)) +
  geom_label_repel(aes(label = round(prop_correct, 2)), size = 2.5) +
  #ggtitle(str_c("Participant = ", participant_sample)) +
  theme_bw() +
  labs(x = expr(x[1]), y = expr(x[2]))

tbl_summary %>% head(12)
tbl_naive2 %>% grouped_agg(category, c(d1i_z, d2i_z)) %>%
  mutate(
    sd_d1i_z = se_d1i_z * sqrt(n),
    sd_d2i_z = se_d2i_z * sqrt(n)
  )


