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

# compute pairwise distances
m_distances_x1 <- map(1:nrow(tbl_sample), distance_1d, tbl_sample, "d1i") %>% unlist() %>%
  matrix(byrow = TRUE, nrow = nrow(tbl_sample), ncol = nrow(tbl_sample))
m_distances_x2 <- map(1:nrow(tbl_sample), distance_1d, tbl_sample, "d2i") %>% unlist() %>%
  matrix(byrow = TRUE, nrow = nrow(tbl_sample), ncol = nrow(tbl_sample))




