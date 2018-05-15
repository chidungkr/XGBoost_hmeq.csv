

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

# Tham số tối ưu: 
xgb_Tree2$bestTune 

# Thiết lập một loạt các tham số tinh chỉnh của xbgTree : 

tune_grid <- expand.grid(
  nrounds = seq(200, 300, by = 50), 
  max_depth = 4:7,
  eta = 0.3,
  gamma = 0, 
  colsample_bytree = 0.6, 
  min_child_weight = 1, 
  subsample = seq(0.1, 0.9, by = 0.1))


# Chạy XGB với lưới tham số tinh chỉnh ở trên: 
set.seed(1)
xgb_Tree3 <- train(BAD ~ ., 
                   data = train_df, 
                   method = "xgbTree", 
                   metric = "AUC", 
                   tuneGrid = tune_grid, 
                   trControl = train.control)

# Tham số tối ưu: 
xgb_Tree3$bestTune

# So sánh 14 tiêu chí đánh giá chất lượng mô hình phân loại
# và thấy rằng XGB3 là tốt nhất nếu căn cứ vào Accuracy: 

df_results <- bind_rows(xgb_Tree1$resample %>% mutate(Model = "XGB1"), 
                        xgb_Tree2$resample %>% mutate(Model = "XGB2"), 
                        xgb_Tree3$resample %>% mutate(Model = "XGB3"))


theme_set(theme_minimal())

df_results %>% 
  select(-Resample) %>% 
  gather(a, b, -Model) %>% 
  ggplot(aes(Model, b, color = Model, fill = Model)) + 
  geom_boxplot(alpha = 0.3, show.legend = FALSE) + 
  facet_wrap(~ a, scales = "free") + 
  labs(x = NULL, y = NULL)

# Tính toán cụ thể: 
df_results %>% 
  group_by(Model) %>% 
  summarise_each(funs(median, mean, min, max, n()), Accuracy) %>% 
  arrange(-Accuracy_median)


df_results %>% 
  group_by(Model) %>% 
  summarise_each(funs(median, mean, min, max, n()), AUC) %>% 
  arrange(-AUC_median)


# Viết hàm tính CM: 

my_cm <- function(model, test_df) {
  pred <- predict(model, test_df)
  confusionMatrix(test_df$BAD, pred)
}


# Có dấu hiệu cho thấy XGB2 là tốt nhất nếu mục tiêu là vì lợi nhuận: 
my_cm(xgb_Tree1, test_df = test_df %>% group_by(BAD) %>% sample_n(500) %>% ungroup())
my_cm(xgb_Tree2, test_df = test_df %>% group_by(BAD) %>% sample_n(500) %>% ungroup())
my_cm(xgb_Tree3, test_df = test_df %>% group_by(BAD) %>% sample_n(500) %>% ungroup())

# Viết hàm tính BG - GB: 

my_results <- function(model, input_df) {
  pred <- predict(model, input_df)
  m <- confusionMatrix(input_df$BAD, pred)
  m$table %>% 
    as.vector() %>% 
    matrix(ncol = 4) %>% 
    as.data.frame() %>% 
    rename(BB = V1, GB = V2, BG = V3, GG = V4) %>% 
    return()
  
}

# Cho XGB1 đến XGB3: 

lapply(1:1000, function(i) {
  set.seed(i)
  my_results(xgb_Tree1,  test_df %>% group_by(BAD) %>% sample_n(500) %>% ungroup())
  
}) -> m1

lapply(1:1000, function(i) {
  set.seed(i)
  my_results(xgb_Tree2,  test_df %>% group_by(BAD) %>% sample_n(500) %>% ungroup())
  
}) -> m2

lapply(1:1000, function(i) {
  set.seed(i)
  my_results(xgb_Tree3,  test_df %>% group_by(BAD) %>% sample_n(500) %>% ungroup())
  
}) -> m3


# Hàm chuyển hóa về DF: 
convert_to_df <- function(x) {
  do.call("bind_rows", x)
}


# Sử dụng hàm: 
df1 <- convert_to_df(m1) %>% mutate(Model = "XGB1")
df2 <- convert_to_df(m2) %>% mutate(Model = "XGB2")
df3 <- convert_to_df(m3) %>% mutate(Model = "XGB3")

# Xếp nhầm hồ sơ Xấu thành tốt thì XGB2 thấp nhất: 
df1 %>% pull(BG) %>% summary()
df2 %>% pull(BG) %>% summary()
df3 %>% pull(BG) %>% summary()

# XGB3 xếp đúng tốt thành tốt cao nhất: 
df1 %>% pull(GG) %>% summary()
df2 %>% pull(GG) %>% summary()
df3 %>% pull(GG) %>% summary()




