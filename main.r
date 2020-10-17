library(autofeat)
#dataf <- read.csv("data/gina.csv")

#dataf <- read.csv("data/lymphoma_2classes.csv")
#dataf <- read.csv("data/banknote.csv")
#dataf <- read.csv("data/micro-mass.csv")
#dataf <- read.csv("data/dbworld-bodies.csv")
#dataf <- read.csv("data/nomao.csv")
dataf <- read.csv("input.csv")



y <- factor(dataf$Class)
X <- data.matrix(dataf[ , !(names(dataf) %in% c("Class"))])

i <- sample(1:nrow(X), round(0.3 * nrow(X)))
X_train <- X[i,]
y_train <- y[i]
X_valid <- X[-i,]
y_valid <- y[-i]


#SAFE(x,y,x,y,alpha=0.00001,theta=1)
res <- SAFE(X_train, y_train, X_valid, y_valid)
#res <- SAFE(X, y, X, y)

new_X <- cbind(res$X_train, class = y_train)
write.csv(new_X, "test.csv" ,row.names=FALSE)


print("Done!")



