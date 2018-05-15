

# Load dữ liệu: 

rm(list = ls())
library(tidyverse)
library(magrittr)

hmeq <- read.csv("D:/Teaching/data_science_banking/hmeq/hmeq.csv")

# Viết một số hàm xử lí số liệu thiếu: 
thay_na_mean <- function(x) {
  tb <- mean(x, na.rm = TRUE)
  x[is.na(x)] <- tb
  return(x)
}


name_job <- function(x) {
  x %<>% as.character()
  ELSE <- TRUE
  quan_tam <- c("Mgr", "Office", "Other", "ProfExe", "Sales", "Self")
  case_when(!x %in% quan_tam ~ "Other", 
            ELSE ~ x)
}


name_reason <- function(x) {
  ELSE <- TRUE
  x %<>% as.character()
  case_when(!x %in% c("DebtCon", "HomeImp") ~ "Unknown", 
            ELSE ~ x)
}

label_rename <- function(x) {
  case_when(x == 1 ~ "BAD", 
            x == 0 ~ "GOOD")
}


# Xử lí số liệu thiếu và dán nhãn lại: 
hmeq_proce <- hmeq %>% 
  mutate_if(is.numeric, thay_na_mean) %>% 
  mutate_at("REASON", name_reason) %>% 
  mutate_at("JOB", name_job) %>% 
  mutate(BAD = label_rename(BAD)) %>% 
  mutate_if(is.character, as.factor)

# Kiểm tra: 
hmeq_proce %>% summarise_all(class)


# Thực hiện phân chia dữ liệu: 
set.seed(1)
train_df <- hmeq_proce %>% 
  group_by(BAD) %>% 
  sample_frac(0.5) %>% 
  ungroup()

test_df <- dplyr::setdiff(hmeq_proce, train_df)

# Các tham số có thể tinh chỉnh của mô hình: 
library(caret)
modelLookup(model = "xgbTree")

# Thiết lập chế độ đánh giá mô hình: 

set.seed(1)
train.control <- trainControl(method = "repeatedcv", 
                              number = 10,
                              repeats = 3, 
                              classProbs = TRUE,
                              allowParallel = TRUE, 
                              summaryFunction = multiClassSummary)

# xgbTree mặc định: 

library(doParallel)
registerDoParallel(6)
getDoParWorkers()

set.seed(2)
xgb_Tree1 <- train(BAD ~ ., 
                   data = train_df, 
                   method = "xgbTree", 
                   metric = "Accuracy", 
                   trControl = train.control)

# Các tham số tối ưu của mô hình: 
xgb_Tree1$bestTune 


# XGB khi tiêu chí tối ưu tham số là AUC: 
set.seed(2)
xgb_Tree2 <- train(BAD ~ ., 
                   data = train_df, 
                   method = "xgbTree", 
                   metric = "AUC", 
                   tuneLength = 5, 
                   trControl = train.control)

xgb_Tree2$bestTune 

# So sánh 14 tiêu chí đánh giá chất lượng mô hình phân loại
# và thấy rằng XGB2 - tức tối ưu hóa tham số theo AUC là tốt hơn: 
df_results <- bind_rows(xgb_Tree1$resample %>% mutate(Model = "XGB1"), 
                        xgb_Tree2$resample %>% mutate(Model = "XGB2"))


theme_set(theme_minimal())
df_results %>% 
  select(-Resample) %>% 
  gather(a, b, -Model) %>% 
  ggplot(aes(Model, b, color = Model, fill = Model)) + 
  geom_boxplot(alpha = 0.3, show.legend = FALSE) + 
  facet_wrap(~ a, scales = "free") + 
  labs(x = NULL, y = NULL)


# Thiết lập một loạt các tham số tinh chỉnh của xbgTree : 


tune_grid <- expand.grid(
  nrounds = seq(200, 300, by = 50), 
  max_depth = 4:6,
  eta = seq(0.2, 0.5, by = 0.1),
  gamma = seq(0, 0.1, by = 0.025), 
  colsample_bytree = seq(0.5, 0.8, by = 0.1), 
  min_child_weight = seq(1, 3, by = 1), 
  subsample = 1:3)


# có lẽ cần 24h để chạy mô hình: 
set.seed(1)
xgb_Tree3 <- train(BAD ~ ., 
                   data = train_df, 
                   method = "xgbTree", 
                   metric = "AUC", 
                   tuneGrid = tune_grid, 
                   trControl = train.control)

xgb_Tree3







