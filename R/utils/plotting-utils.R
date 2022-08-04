plot_average_categorization_accuracy <- function(tbl_df, title) {
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
