setwd("titanic")
train<-read.csv("train.csv")
test<-read.csv("test.csv")
#gm<-read.csv("gendermodel.csv")
library(dplyr)
library(tidyr)
library(neuralnet)

train %>% sapply(function(x){sum(is.na(x))})

newage<-function(age,class,sex){if(is.na(age)){
                mean(train[train$Sex==sex &
                                   train$Pclass==class,]$Age,na.rm=TRUE)}
                else age}

train$newage<- mapply(newage, train$Age, train$Pclass, train$Sex)

tr1<-train %>%
        mutate(lname=sapply(as.character(Name),nchar)) %>%
        mutate(ageknown=is.na(Age)) %>%
        dplyr::select(-Ticket,-Cabin,-Name,-Age)

bins<-tr1 %>% # piped functions follow
        
        # make it narrow, don't touch numeric variables and IDs
        gather(catnames,catvalues,-PassengerId,-Survived,-newage,-Fare,-SibSp,
               -Parch,-lname,-ageknown) %>%
        
        # make single column out of them
        unite(newfactor,catnames,catvalues,sep=".") %>%
        
        # add a new column - it's "1" for every record
        mutate( is = 1) %>%
        
        # create a column from each factor, and where there's no record, add "0"
        spread(newfactor, is, fill = 0)

seed<-2
bins.nn<-function(df,rep=1,hidden=c(1),threshold=0.1) {
        set.seed(seed)
        nn.obj<-neuralnet(Survived ~ SibSp+ Parch+ Fare+ newage+ lname+ ageknown+ Embarked.+ Embarked.C+ Embarked.Q+ Embarked.S+ Pclass.1+ Pclass.2+ Pclass.3+ Sex.female+ Sex.male,
                          data=df,
                          hidden=hidden,
                          lifesign="full",
                          lifesign.step=2000,
                          threshold=threshold,
                          rep=rep)
        return(nn.obj)}

qualify<-function(real,guess){
        check<-table(real,guess)
        good.ones<-check[1,1]+check[2,2]
        bad.ones<-check[1,2]+check[2,1]
        paste0(as.character(round(100*good.ones/(good.ones+bad.ones))),'%')
}

n.full<-bins.nn(bins,rep=1,hidden=c(5),threshold=0.02)

nr<-dim(bins)[1] # number of observations
share<-0.8 # this is our 80% parameter
set.seed(seed)
trainset<-sample.int(nr,round(share*nr))

trainers<-bins[trainset,]
testers<-bins[-trainset,]

n.testers<-bins.nn(trainers,rep=1,hidden=c(5),threshold=0.02)

nfeat<-dim(bins)[2] 
res.testers<-neuralnet::compute(n.testers,testers[,3:nfeat]) # Your columns instead of 3
qualify(round(res.testers$net.result),testers$Survived)

# OK, 85%
# retrain to full
n.full<-bins.nn(bins,rep=10,hidden=c(5),threshold=0.02)

##### Now test

test$newage<- mapply(newage, test$Age, test$Pclass, test$Sex)

ts1<-test %>%
        mutate(lname=sapply(as.character(Name),nchar)) %>%
        mutate(ageknown=is.na(Age)) %>%
        dplyr::select(-Ticket,-Cabin,-Name,-Age)

bins.test<-ts1 %>% # piped functions follow
        
        # make it narrow, don't touch numeric variables and IDs
        gather(catnames,catvalues,-PassengerId,-newage,-Fare,-SibSp,
               -Parch,-lname,-ageknown) %>%
        
        # make single column out of them
        unite(newfactor,catnames,catvalues,sep=".") %>%
        
        # add a new column - it's "1" for every record
        mutate( is = 1) %>%
        
        # create a column from each factor, and where there's no record, add "0"
        spread(newfactor, is, fill = 0)

nfeat.test<-dim(bins.test)[2] 

# wrong columns count - because in train set there were two spaces in Embarked,
# let's drop these lines
bins1<-bins[bins$Embarked.==0,] %>% dplyr::select(-Embarked.)

bins.nn1<-function(df,rep=1,hidden=c(1),threshold=0.1) {
        set.seed(seed)
        nn.obj<-neuralnet(Survived ~ SibSp+ Parch+ Fare+ newage+ lname+ ageknown+ Embarked.C+ Embarked.Q+ Embarked.S+ Pclass.1+ Pclass.2+ Pclass.3+ Sex.female+ Sex.male,
                          data=df,
                          hidden=hidden,
                          lifesign="full",
                          lifesign.step=2000,
                          threshold=threshold,
                          rep=rep)
        return(nn.obj)}

n.full<-bins.nn1(bins1,rep=3,hidden=c(5),threshold=0.02)
res<-neuralnet::compute(n.full,bins.test[,2:nfeat.test])
upload<-round(res$net.result)
names(upload)<-c("Survived")
upload1<-data.frame(cbind(bins.test$PassengerId,upload))
names(upload1)<-c("PassengerId","Survived")
write.csv(upload1,file="res1.csv",row.names=FALSE)
# Got NA and two "2"
