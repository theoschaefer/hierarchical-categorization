library(cmdstanr)
library(rutils)
library(tidyverse)

dir_home_grown <- c("R/utils/ellipse-utils.R", "R/utils/utils.R")
walk(dir_home_grown, source)

check_cmdstan_toolchain(fix = TRUE, quiet = TRUE)
check_cmdstan_toolchain()


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
  n_correct = ceiling(exp(-(abs(x1) + abs(x2))) * n_trials_per_stim),
  n_trials = n_trials_per_stim
)
tbl$n_correct[tbl$category == 1 & tbl$n_correct == 1] <- 3#abs(tbl$n_correct[tbl$category == 1] - max(tbl$n_correct))
ggplot(tbl, aes(x1, x2, group = category)) +
  geom_raster(aes(fill = category, alpha = n_correct)) +
  guides(fill = "none") +
  theme_bw()

is_same_category <- function(i, tbl) {
  # same category gets a 1, different a 2
  abs(as.numeric(tbl$category[i] == tbl$category) - 1) + 1
}
# compute pairwise distances
m_distances_x1 <- map(1:nrow(tbl), distance_1d, tbl, "x1") %>% unlist() %>%
  matrix(byrow = TRUE, nrow = nrow(tbl), ncol = nrow(tbl))
m_distances_x2 <- map(1:nrow(tbl), distance_1d, tbl, "x2") %>% unlist() %>%
  matrix(byrow = TRUE, nrow = nrow(tbl), ncol = nrow(tbl))

# Run Stan Model ----------------------------------------------------------

# NOTE should byrow be true?? 
stan_gcm <- write_stan_file("
data {
  int n_stim;
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
}

transformed parameters {
  array[n_stim, n_stim] real <lower=0,upper=1> s;
  array[n_stim, 2] real <lower=0> sumsim;
  array[n_stim] real <lower=0,upper=1> theta;
  row_vector[2] bs = [b, 1 - b];
  
  // Similarities
  for (i in 1:n_stim){
  sumsim[i, 1] = 0;
  sumsim[i, 2] = 0;
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
mod <- cmdstan_model(stan_gcm)
vars <- mod$variables()
names(vars$data)
mod$exe_file()

l_data <- list(
  n_stim = nrow(tbl), n_trials = tbl$n_trials, n_correct = tbl$n_correct, 
  cat = as.numeric(as.character(tbl$category)),
  d1 = m_distances_x1, d2 = m_distances_x2
)


fit <- mod$sample(data = l_data, iter_sampling = 500, iter_warmup = 1000, chains = 1)
tbl_summary <- fit$summary()
vars_extract <-  c("c", "w", "bs", "theta")
tbl_draws <- fit$draws(variables = vars_extract, format = "df")
tbl_posterior <- tbl_draws %>% 
  select(starts_with("theta"), .chain) %>% colMeans()
  rename(chain = .chain) %>%
  pivot_longer(mu, names_to = "parameter", values_to = "value")
kd <- rutils::estimate_kd(tbl_posterior, "mu")

params_bf <- c("mu")
l <- sd_bfs(tbl_posterior, params_bf, sqrt(2)/4)
#l[[2]]$value[l[[2]]$variable == "thxhi_x"] <- 1
rutils::plot_posterior("mu", tbl_posterior, l[[2]])


tbl$pred_accuracy <- tbl_draws %>% 
  select(starts_with("theta")) %>% colMeans()


ggplot(tbl, aes(x1, x2, group = category)) +
  geom_raster(aes(fill = category, alpha = pred_accuracy)) +
  guides(fill = "none") +
  theme_bw()

tbl_long <- tbl_draws[, c("theta[1]", "theta[15]", "theta[35]", "theta[36]")] %>% 
  pivot_longer(c("theta[1]", "theta[15]", "theta[35]", "theta[36]"), names_to = "param")
ggplot(tbl_long, aes(value, group = param)) +
  geom_density(aes(color = param)) +
  scale_color_brewer(palette = "Set1")


ggplot(tbl, aes(n_correct, pred_means, group = category)) +
  geom_point(aes(color = category)) +
  stat_summary(geom = "line", fun.y = mean)
