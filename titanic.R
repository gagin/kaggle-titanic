# setwd("kaggle-titanic/")

train<-read.csv("train.csv")
test<-read.csv("test.csv")
library(dplyr)
library(tidyr)
library(neuralnet)

# check for NAs
# train %>% sapply(function(x){sum(is.na(x))})

# drop two entries with no Embarked values
train<-train[train$Embarked!="",]

# set NA ages to mean for the class and sex
newage<-function(age,class,sex){if(is.na(age)){
                mean(train[train$Sex==sex &
                                   train$Pclass==class,]$Age,na.rm=TRUE)}
                else age}

train$newage<- mapply(newage, train$Age, train$Pclass, train$Sex)

# make name length a feature
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
##prepare list for neuralnet() call
#cat(paste0(names(bins),sep="+"))
bins.nn<-function(df,rep=1,hidden=c(1),threshold=0.1) {
        set.seed(seed)
        nn.obj<-neuralnet(Survived ~ SibSp+ Parch+ Fare+ newage+ lname+ ageknown+ Embarked.C+ Embarked.Q+ Embarked.S+ Pclass.1+ Pclass.2+ Pclass.3+ Sex.female+ Sex.male,
                          data=df,
                          hidden=hidden,
                          lifesign="full",
                          lifesign.step=2000,
                          threshold=threshold,
                          rep=rep)
        return(nn.obj)}

# clean up results from NAs and 2s
cleanup<-function(vect){
        sapply(vect,function(x){
                if(is.na(x)) 0
                else if(x>0) 1
                else 0})}

# In internal tests, split train set to train/test part and check how
# the selected algorithm works
## BEGIN internal tests
if(FALSE){
        qualify<-function(real,guess){
                check<-table(real,guess)
                good.ones<-check[1,1]+check[2,2]
                bad.ones<-check[1,2]+check[2,1]
                paste0(as.character(round(100*good.ones/(good.ones+bad.ones))),'%')
        }
        
        nr<-dim(bins)[1] # number of observations
        share<-0.8 # this is our 80% parameter
        set.seed(seed)
        trainset<-sample.int(nr,round(share*nr))
        
        trainers<-bins[trainset,]
        testers<-bins[-trainset,]
        
        nfeat<-dim(bins)[2] 
        
        n.testers<-bins.nn(trainers,rep=3,hidden=c(4),threshold=0.25)
        
        res.testers<-neuralnet::compute(n.testers,testers[,3:nfeat])
        qualify(cleanup(round(res.testers$net.result)),testers$Survived)
}
## END internal tests
        
# Train with full train set
n.full<-bins.nn(bins,rep=5,hidden=c(4),threshold=0.25)

##### Now test data

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

res<-neuralnet::compute(n.full,bins.test[,2:nfeat.test])
upload<-cleanup(round(res$net.result))


names(upload)<-c("Survived")
upload1<-data.frame(cbind(bins.test$PassengerId,upload))
names(upload1)<-c("PassengerId","Survived")
write.csv(upload1,file="res.csv",row.names=FALSE,quote=FALSE)

