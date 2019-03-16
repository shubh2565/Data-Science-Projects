




# In this section, we will explore the data through visualization and other methods and make some inference about our data. Further, we will look out for any irreregularities in the dataset, so that we can correct them in the pre-processing stage. First, load libraries and read data from the file.

```{r message=FALSE}
library(data.table) 
library(dplyr)     
library(ggplot2)    
library(caret)      
library(corrplot)  
library(xgboost)   
library(cowplot)
library(gridExtra)
library(dummies)
library(stringr)


train <- read.csv('Train.csv')
test <- read.csv('Test.csv')

str(train)
str(test)
```


#Now, we combine our train and test dataset so that we don't need to do the data cleansing and data manipulation steps twice. After making the desired changes we can split the data again before doing the regression analysis.

test$Item_Outlet_Sales <- NA
data_combined <- rbind(train, test)
dim(data_combined)


#### Univariate Analysis


train_numeric = dplyr::select_if(train, is.numeric)
names(train_numeric)



plot_weight <- ggplot(data_combined) + geom_histogram(aes(Item_Weight), color="black", fill="grey")
plot_visibility <- ggplot(data_combined) + geom_histogram(aes(Item_Visibility), color="black", fill="grey")
plot_mrp <- ggplot(data_combined) + geom_histogram(aes(Item_MRP), color="grey", fill="grey", binwidth = 0.5)
grid.arrange(plot_weight, plot_visibility, plot_mrp, ncol = 2, nrow = 2)

#Now, we will explore the categorical variables to gain some more insights about our dataset.

ggplot(data_combined %>% group_by(Item_Type) %>% summarise(Count = n())) +   geom_bar(aes(Item_Type, Count, , fill = interaction(Item_Type, Count, sep = ": ")), stat = "identity") +  xlab("") +  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))+  ggtitle("Item_Type")

plot_outletSize <- ggplot(data_combined %>% group_by(Outlet_Size) %>% summarise(Count = n())) +   geom_bar(aes(Outlet_Size, Count), stat = "identity", fill = "coral1") +  geom_label(aes(Outlet_Size, Count, label = Count), vjust = 0.5, size =2.5) +  theme(axis.text.x = element_text(angle = 45, hjust = 1))

plot_fatContent <- ggplot(data_combined %>% group_by(Item_Fat_Content) %>% summarise(Count = n())) +   geom_bar(aes(Item_Fat_Content, Count), stat = "identity", fill = "coral1") + geom_label(aes(Item_Fat_Content, Count, label = Count), vjust = 0.5, size = 2.5) + theme(axis.text.x = element_text(angle = 45, hjust = 1))

grid.arrange(plot_outletSize, plot_fatContent, ncol = 2)


#Multivariate Analysis


plot1 <- ggplot(train) + geom_point(aes(Item_Weight, Item_Outlet_Sales), colour = "skyblue", alpha = 0.3) + theme(axis.title = element_text(size = 8.5))
plot2 <- ggplot(train) + geom_point(aes(Item_Visibility, Item_Outlet_Sales), colour = "skyblue", alpha = 0.3) +theme(axis.title = element_text(size = 8.5))
plot3 <- ggplot(train) + geom_point(aes(Item_MRP, Item_Outlet_Sales), colour = "skyblue", alpha = 0.3) + theme(axis.title = element_text(size = 8.5))

second_row_2 = plot_grid(plot2, plot3, ncol = 2)
plot_grid(plot1, second_row_2, nrow = 2)

plot4 <- ggplot(train) + geom_boxplot(aes(Outlet_Identifier, sqrt(Item_Outlet_Sales), fill = Outlet_Type)) + theme_minimal() + theme(axis.text.x = element_text(angle = 90))
plot4



#  Data Pre-processing


data_combined$Item_Fat_Content <-str_replace(str_replace(str_replace(data_combined$Item_Fat_Content,"LF","Low Fat"),"reg","Regular"),"low fat","Low Fat")

table(data_combined$Item_Fat_Content)


# We saw on exploring part, Item_Weight column has null values which can affect the result of anaylsis. Impute missing value by median. We are using median because it is known to be highly robust to outliers. Moreover, for this problem, our evaluation metric is RMSE which is also highly affected by outliers. Hence, median is better in this case.

sum(is.na(data_combined$Item_Weight))

data_combined$Item_Weight[is.na(data_combined$Item_Weight)] <- 
median(data_combined$Item_Weight, na.rm = TRUE)
sum(is.na(data_combined$Item_Weight))

#Let’s take up Item_Visibility. On exploration part above, we saw item visibility has zero value also,which is practically not possible. Hence, we’ll consider it as a missing value and once again make the imputation using median.


data_combined$Item_Visibility <- ifelse(data_combined$Item_Visibility == 0,
median(data_combined$Item_Visibility),
data_combined$Item_Visibility)
ggplot(data_combined) + geom_histogram(aes(Item_Visibility), bins = 100, color="red", fill="white")



#We need to mutate new columns for more meaningful data. First, we evaluate Item_Identifier column because we discovered Item_Identifier column has special codes to recognize the type of item when we tried to understand data. We will use first two letters (DR = Drink, FD = Food, NC = Non-Consumable). Secondly, we generate new Outlet_Age column from Outlet_Establishment_Year. And, we will also change the values of Item_Fat_Content wherever Item_category is ‘NC’ because non-consumable items cannot have any fat content.

data_combined <- data_combined %>% 
mutate(Item_Category = substr(Item_Identifier, 1, 2),
Outlet_Age = 2013 - Outlet_Establishment_Year)

table(data_combined$Item_Category)

data_combined$Item_Fat_Content[data_combined$Item_Category == "NC"] = "Non-Edible" 

table(data_combined$Item_Fat_Content)


#Since, machine learning and data science algorithms work only on numerical variables, we need to do One Hot Encoding for our categorical data.

data_combined <- dummy.data.frame(data_combined, names = c('Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Category', 'Outlet_Identifier'), sep ='_')
data_combined <- subset(data_combined, select = -c(Item_Identifier, Item_Type, Outlet_Establishment_Year))
str(data_combined)

# Now, remove skewness from the variable Item_Visibility.

data_combined$Item_Visibility <- sqrt(data_combined$Item_Visibility)
ggplot(data_combined) + geom_histogram(aes(Item_Visibility), bins = 100, color="red", fill="white")


# Now, we scale our numerical predeictors. We use Z-score normalization.

data_combined$Item_Weight <- scale(data_combined$Item_Weight, center= TRUE, scale=TRUE)
data_combined$Item_Visibility <- scale(data_combined$Item_Visibility, center= TRUE, scale=TRUE)
data_combined$Item_MRP <- scale(data_combined$Item_MRP, center= TRUE, scale=TRUE)
data_combined$Outlet_Age <- scale(data_combined$Outlet_Age, center= TRUE, scale=TRUE)

str(data_combined)


# Now, we split the data_combined back into train and test data for building our model.

train <- data_combined[1:nrow(train), ]
test <- data_combined[(nrow(train) + 1):nrow(data_combined), ]
test <- subset(test, select = -c(Item_Outlet_Sales))

dim(train)
dim(test)


# Now, we check for the correlation among the variables to decide whether we need to reduce dimensions of our dataset or not.

corMatrix <- cor(train[, -35])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
tl.cex = 0.6, tl.col = 'black')



# Model Building


# Linear Regression


# building the model
set.seed(1000)
linear_reg_mod = lm(Item_Outlet_Sales ~ ., data = train)
summary(linear_reg_mod)


# Just keeping the significant variables.

linear_reg_mod = lm(Item_Outlet_Sales ~ Item_MRP + Outlet_Identifier_OUT010 + Outlet_Identifier_OUT018 + Outlet_Identifier_OUT019 + Outlet_Identifier_OUT027 + Outlet_Identifier_OUT045, data = train)
summary(linear_reg_mod)

# making predictions on test data
prediction = predict(linear_reg_mod, test)


# Ridge Regression

set.seed(1357)
my_control <- trainControl(method="cv", number=5)
Grid <- expand.grid(alpha = 0, lambda = seq(0.001,0.1,by = 0.0002))
ridge_linear_reg_mod <- train(Item_Outlet_Sales ~ Item_MRP + Outlet_Identifier_OUT010 + Outlet_Identifier_OUT018 + Outlet_Identifier_OUT019 + Outlet_Identifier_OUT027 + Outlet_Identifier_OUT045, data = train, method='glmnet', trControl= my_control, tuneGrid = Grid)

# making predictions on test data
prediction = predict(ridge_linear_reg_mod, test)



# Random Forest Regression


set.seed(1237) 
my_control <- trainControl(method="cv", number=5)
tgrid = expand.grid(.mtry = c(2:6),
.splitrule = "variance",
.min.node.size = c(10,15,20))
rf_mod <- train(Item_Outlet_Sales ~ Item_MRP + Outlet_Identifier_OUT010 + 
Outlet_Identifier_OUT018 + Outlet_Identifier_OUT019 +
Outlet_Identifier_OUT027 + Outlet_Identifier_OUT045, data = train,
method='ranger',
tuneGrid = tgrid,
trControl= my_control,
num.trees = 500,
importance = "permutation")

prediction = predict(rf_mod, test)
write.csv(prediction, "Linear_Reg_submit.csv", row.names = F)

plot(rf_mod)
plot(varImp(rf_mod))
