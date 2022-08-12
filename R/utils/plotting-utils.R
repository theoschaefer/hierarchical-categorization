plot_average_categorization_accuracy <- function(tbl_df, title) {
  #' scatter plot with proportion correct by x1 and x2
  #' 
  #' @description size of points is plotted according to categorization accuracy
  #' 
  #' @param tbl_df tbl df with n trials and n correct per x1-x2 combination
  #' @param title title of the plot
  #' @return ggplot object
  #' 
  
  tbl_df %>% group_by(d1i, d2i, category) %>% 
    summarize(acc_mn = mean(accuracy)) %>%
    ungroup() %>%
    ggplot(aes(d1i, d2i, group = category)) +
    geom_point(aes(color = category, size = acc_mn), show.legend = FALSE) +
    geom_label_repel(aes(label = round(acc_mn, 2)), size = 2.5, show.legend = FALSE) +
    scale_color_brewer(palette = "Set1") +
    scale_x_continuous(breaks = seq(2, 10, by = 2)) +
    scale_y_continuous(breaks = seq(2, 10, by = 2)) +
    theme_bw() +
    labs(x = expr(x[1]), y = expr(x[2]), title = title)
}


plot_item_thetas <- function(tbl_df, title) {
  #' scatter plot of posterior theta means against proportion correct per item
  #' 
  #' @description adds a label with the correlation between posterior means 
  #' and empirical data (i.e., proportion correct)
  #' 
  #' @param tbl_df tbl df with n trials and n correct per x1-x2 combination
  #' @param title title of the plot
  #' @return ggplot object
  #' 

  tbl_cor <- tibble(
    cor = cor(tbl_df$prop_correct, tbl_df$pred_theta)
  )
  
  ggplot(tbl_df, aes(pred_theta, prop_correct)) +
    geom_point() +
    geom_abline() +
    geom_label(data = tbl_cor, aes(label = str_c("r = ", round(cor, 2)), x = .75, y = .2)) +
    theme_bw() +
    labs(x = "Predicted Theta", y = "Proportion Correct", title = title)
  
}


plot_proportion_responses <- function(tbl_df, facet_by_response = FALSE) {
  #' scatter plot of stimuli in feature space
  #' 
  #' @description filled by true category, faceted by response, 
  #' and sized by proportion responses
  #' 
  #' @param tbl_df tbl df with x vals in z space, stim_id, category, response,
  #' and proportion responses
  #' @param facet_by_response facets by given responses; default to FALSE
  #' @return ggplot object
  #' 
  
  pl <- ggplot(tbl_df, aes(d1i_z, d2i_z)) +
    geom_point(aes(size = prop_responses, color = category)) +
    geom_label_repel(aes(label = str_c(stim_id, "=", round(prop_responses, 2))), size = 2.5) +
    ggtitle(str_c("Participant = ", participant_sample)) +
    scale_color_brewer(palette = "Set1", name = "True Category") +
    scale_size_continuous(guide = "none") +
    theme_bw() +
    labs(x = expr(x[1]), y = expr(x[2]))
  if(facet_by_response) {
    pl + facet_wrap(~ response)
  } else {pl}
}
