library(data.table) # 用于数据操作
library(mlr3verse) # 加载 mlr3 生态系统

library("tidyverse")
library("ggplot2")

set.seed(212) # 设置随机数种子，保证结果可复现

# 载入数据
# 使用 fread 载入 bank_personal_loan.csv 数据
bank_data <- fread("bank_personal_loan.csv")
bank_data <- subset(bank_data, select = -ZIP.Code)
# 将指定的列转换为因子类型
bank_data$CreditCard <- as.factor(bank_data$CreditCard)
bank_data$Online <- as.factor(bank_data$Online)
bank_data$CD.Account <- as.factor(bank_data$CD.Account)
bank_data$Securities.Account <- as.factor(bank_data$Securities.Account)
bank_data$Education <- as.factor(bank_data$Education)
#bank_data$ZIP.Code <- as.factor(bank_data$ZIP.Code)
bank_data$Personal.Loan <- as.factor(bank_data$Personal.Loan)
bank_data$Family <- factor(bank_data$Family, ordered = TRUE)


skimr::skim(bank_data)

DataExplorer::plot_bar(bank_data, ncol = 3)
DataExplorer::plot_histogram(bank_data, ncol = 3)
DataExplorer::plot_boxplot(bank_data, by = "Personal.Loan", ncol = 3)



bank_task <- TaskClassif$new(id = "BankPersonalLoan",
                             backend = as.data.table(bank_data),
                             target = "Personal.Loan", # 确保这是你的目标变量
                             positive = "1") # 根据你的数据情况调整 positive 参数

cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(bank_task)

# 定义学习器
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.0115, id = "cartcp")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")

lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")
lrn_lda      <- lrn("classif.lda", predict_type = "prob")


pl_xgb <- po("encode") %>>%
  po(lrn_xgboost)


# 现在像往常一样进行拟合... 我们可以将它添加到我们的基准测试集中
res <- benchmark(data.table(
  task       = list(bank_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp,
                    lrn_ranger,
                    pl_xgb,
                    lrn_log_reg,
                    lrn_lda),
  resampling = list(cv5)
), store_models = TRUE)


res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))



#图一
library(ggplot2)
# 假设的模型性能数据框
model_performance <- data.frame(
  Model = c("Baseline", "Decision Tree", "Decision Tree (cartcp)", "Random Forest", 
            "XGBoost", "Logistic Regression", "LDA", "Super Learner"),
  CE = c(0.096, 0.017, 0.0184, 0.014, 0.015, 0.0404, 0.0536, 0.0124)  # 分类错误率
)
# 绘制性能对比图
ggplot(model_performance, aes(x = reorder(Model, CE), y = CE, fill = Model)) +
  geom_col() +
  theme_minimal() +
  labs(title = "Model Classification Error Comparison", x = "Model", y = "Classification Error (CE)") +
  coord_flip() + # 横向条形图更易读
  scale_fill_brewer(palette = "Set3") +
  theme(legend.position = "none")

#图五
lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)

res_cart_cv <- resample(bank_task, lrn_cart_cv, cv5, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[5]]$model)


#
#超级学习器

bank_task <- TaskClassif$new(id = "BankPersonalLoan",
                             backend = as.data.table(bank_data),
                             target = "Personal.Loan", # 确保这是你的目标变量
                             positive = "1") # 根据你的数据情况调整 positive 参数


#定义用于误差估计的重采样策略
cv5 <- rsmp("cv", folds = 10)
cv5$instantiate(bank_task)

# 定义一系列基础学习器
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.044, id = "cartcp")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")

lrn_lda      <- lrn("classif.lda", predict_type = "prob")

# 定义超级学习器
lrnsp_log_reg <- lrn("classif.log_reg", predict_type = "prob", id = "super")

# 处理缺失值的流水线
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

# 类别变量编码流水线
pl_factor <- po("encode")

# 定义完整流水线
spr_lrn <- gunion(list(
  # 第一组无需修改输入的学习器
  gunion(list(
    po("learner_cv", lrn_baseline),
    po("learner_cv", lrn_cart),
    po("learner_cv", lrn_cart_cp)
  )),
  # 第二组需要特殊处理缺失值的学习器
  pl_missing %>>%
    gunion(list(
      po("learner_cv", lrn_ranger),
      po("learner_cv", lrn_log_reg),
      po("learner_cv", lrn_lda),
      po("nop") # 这将处理过缺失值的原始特征传递给超级学习器
    )),
  # 第三组需要因子编码的学习器
  pl_factor %>>%
    po("learner_cv", lrn_xgboost)
)) %>>%
  po("featureunion") %>>%
  po(lrnsp_log_reg)


# 将整个流水线转换为一个可用于 resample() 的学习器对象
graph_learner <- GraphLearner$new(spr_lrn)

# 使用 graph_learner 进行交叉验证
res_spr <- resample(bank_task, graph_learner, cv5, store_models = TRUE)

# 聚合结果
res_spr$aggregate(msr("classif.ce"))

# 这个图展示了学习流水线的结构
spr_lrn$plot()

# 最后拟合基础学习器和超级学习器，并评估性能
res_spr <- resample(bank_task, spr_lrn, cv5, store_models = TRUE)
res_spr$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.fpr"),
                       msr("classif.fnr")))


# 性能评估
performance <- res_spr$score()

# ROC曲线和AUC
pred <- res_spr$prediction()
roc_curve <- pROC::roc(pred$truth, pred$prob[,2])
auc_value <- pROC::auc(roc_curve)

# 绘制ROC曲线
pROC::plot.roc(roc_curve, print.auc = TRUE)

# 先确保prediction是正确的格式
predictions <- res_spr$prediction()

# 使用yardstick包计算性能指标
library(yardstick)

# 创建一个空的数据框来存储结果
performance_table <- data.frame(
  ACC = accuracy_vec(predictions$truth, predictions$response),
  F1 = f_meas_vec(predictions$truth, predictions$response),
  Precision = precision_vec(predictions$truth, predictions$response),
  Recall = recall_vec(predictions$truth, predictions$response)
)

# 查看性能指标表格
print(performance_table)

# 计算平均指标
average_performance <- colMeans(performance_table, na.rm = TRUE)
print(average_performance)


# 安装和加载calibrate包
#install.packages("calibrate")
library(calibrate)


# 创建数据框
df <- data.frame(predictions = predictions$response, truth = predictions$truth)

# 绘制散点图
plot(df$predictions, df$truth, xlab = "Predicted Probability", ylab = "Actual Probability", main = "Calibration Curve")
abline(a = 0, b = 1, col = "red")  # 添加对角线，表示完美校准



str(res_spr$prediction())





#图4
# Create data frame
results <- data.frame(
  'CV Folds' = c(3, 5, 7, 10),
  'CE' = c(0.014000441, 0.012, 0.011801644, 0.0108),
  'ACC' = c(0.985999559, 0.988, 0.988198356, 0.9892),
  'FPR' = c(0.005974041, 0.005102642, 0.004876919, 0.00441939),
  'FNR' = c(0.089650963, 0.078902006, 0.077295092, 0.07037594)
)

results




# 准备数据
cv_folds <- c(3, 5, 7, 10)
classif_ce <- c(0.014000441, 0.012, 0.011801644, 0.0108)
classif_acc <- c(0.985999559, 0.988, 0.988198356, 0.9892)
classif_fpr <- c(0.005974041, 0.005102642, 0.004876919, 0.00441939)
classif_fnr <- c(0.089650963, 0.078902006, 0.077295092, 0.07037594)

# 创建数据框
results <- data.frame(cv_folds, classif_ce, classif_acc, classif_fpr, classif_fnr)
# 使用ggplot2绘制折线图
library(ggplot2)
ggplot(results, aes(x = cv_folds)) +
  geom_line(aes(y = classif_ce, color = "classif.ce")) +
  geom_line(aes(y = classif_acc, color = "classif.acc")) +
  geom_line(aes(y = classif_fpr, color = "classif.fpr")) +
  geom_line(aes(y = classif_fnr, color = "classif.fnr")) +
  scale_color_manual(values = c("classif.ce" = "red", "classif.acc" = "green",
                                "classif.fpr" = "blue", "classif.fnr" = "purple")) +
  labs(title = "Model Performance Across Different CV Folds", x = "CV Folds", y = "Metric Value") +
  theme_minimal() +
  guides(color = guide_legend(title = "Metrics"))

library(ggplot2)
# 使用ggplot2仅绘制classif.ce的折线图
ggplot(results, aes(x = cv_folds, y = classif_ce)) +
  geom_line(color = "red") +
  geom_point(color = "red") +
  labs(title = "Classification Error Across Different CV Folds",
       x = "CV Folds",
       y = "Classification Error (classif.ce)") +
  theme_minimal()



#图三
#deepcv
library(keras)
library(data.table)
library(caret) # 加载caret包以使用createFolds函数

# 读取数据
bank_data <- fread("bank_personal_loan.csv")
bank_data <- subset(bank_data, select = -ZIP.Code)

# 准备特征和目标变量
features <- as.matrix(bank_data[, .SD, .SDcols = !names(bank_data) %in% c("Personal.Loan")])
target <- as.numeric(bank_data$Personal.Loan)  # 确保目标变量是数值型

# 生成10折交叉验证的索引
set.seed(123)  # 确保结果可重现
folds <- createFolds(target, k = 10)

# 存储交叉验证的结果
cv_results <- list()

# 进行10折交叉验证
for(i in 1:3) {
  # 分割训练集和测试集
  test_indices <- folds[[i]]
  train_indices <- setdiff(1:nrow(features), test_indices)
  
  bank_train_x <- features[train_indices, ]
  bank_train_y <- target[train_indices]
  bank_test_x <- features[test_indices, ]
  bank_test_y <- target[test_indices]
  
  # 构建模型
  deep.net <- keras_model_sequential() %>%
    layer_dense(units = 32, activation = "relu", input_shape = c(ncol(bank_train_x))) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.4) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.4) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.4) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.4) %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  # 编译模型
  deep.net %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(),
    metrics = c("accuracy")
  )
  
  # 训练模型
  history <- deep.net %>% fit(
    bank_train_x, bank_train_y,
    epochs = 50, batch_size = 32,
    validation_split = 0.2, # 使用20%的训练数据作为验证集
    verbose = 2
  )
  
  # 测试模型
  scores <- deep.net %>% evaluate(bank_test_x, bank_test_y, verbose = 0)
  cv_results[[i]] <- scores
}

#print(cv_results[[1]])
# 提取每次迭代的准确率并计算平均值
accuracy_values <- sapply(cv_results, function(result) result["accuracy"])
mean_accuracy <- mean(accuracy_values)

# 打印平均准确率
cat("3-Fold CV Mean Accuracy:", mean_accuracy, "\n")



# 现有的表格数据
results_df <- data.frame(
  Model = c("Baseline", "Decision Tree", "Decision Tree (cartcp)", "Random Forest", 
            "XGBoost", "Logistic Regression", "LDA", "Deep Learning","Super Learner"),
  CE = c(0.0960, 0.0162, 0.0170, 0.0130, 0.0156, 0.0404, 0.0540, NA,0.0124),
  ACC = c(0.9040, 0.9838, 0.9830, 0.9870, 0.9844, 0.9596, 0.9460, 0.9633,0.9876),
  FPR = c(0.000000000, 0.006199478, 0.006199478, 0.002653474, 
          0.005312114, 0.011290106, 0.019016045, NA,0.004891011),
  FNR = c(1.0000000, 0.1095422, 0.1189540, 0.1104535, 
          0.1119121, 0.3144862, 0.3831577, NA,0.085131502)
)

# 加入深度学习的准确率，我们没有其他指标，所以将它们设为NA
# 注意，实际操作中你可能需要计算出这些指标
results_df$ACC[8] <- 0.966  # 假设深度学习的准确率为0.966
results_df$CE[8] <- 1 - results_df$ACC[8]  # 计算分类错误率
results_df$FPR[8] <- NA  # FPR暂无
results_df$FNR[8] <- NA  # FNR暂无

# 打印更新后的表格
print(results_df)




