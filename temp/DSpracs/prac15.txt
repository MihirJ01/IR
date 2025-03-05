# 1 Scattered Diagram (ScatterPlot):
  
  #simple scatterplot from mtcars dataset
  attach(mtcars)
plot(wt,mpg,main="ScatterPlot Example",xlab="Car Weight",ylab="Miles Per Gallon",pch=19)

#Pie Chart:

# Pie Chart with percentages
slices <- c(20,20,30,30)
label <- c('INDIA','US','DUBAI','FRANCE')
pct <- round(slices/sum(slices)*100)
label <- paste(label,pct)
#add percents to labels
label <- paste(label,"%",sep="")
#ad % to labels
pie(slices,labels = label,col=rainbow(length(label)),main="Pie Chart of Countries")

#Box Plot:

#Box Plot
#Boxplot of MPG by car cylinders
boxplot(mpg ~ cyl,data = mtcars,main="Car Milage Data",
        xlab="Number of Cylinders",ylab="Miles per Gallon")