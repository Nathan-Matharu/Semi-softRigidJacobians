library(dplyr)
library(data.table)
setwd("/media/djtface/Chonky/Classes of milk/BME504/finalProject/AllDataCSV") #this should be changed based on the file system

inputFile <- "allSoftFingerexcXYdifDist1cm.csv" #this should be changed for each finger
outputFile <- "1cmPlots.pdf"

data <- read.csv(inputFile)
#print(data)

#order based on excursion 1
exc1 <- data[order(data$EX.T1), ]
#print(exc1)

#order based on Excursion 2
exc2 <- data[order(data$EX.T2), ]
#print(exc2)

#order based on Excursion 3
exc3 <- data[order(data$EX.T3), ]
#print(exc3)

#order based on Excursion 4
exc4 <- data[order(data$EX.T4), ]
#print(exc4)

pdf(outputFile)

#Based on Distance
exc1_dist <- select(exc1, 'EX.T1', 'Dist')
#print(exc1_dist)
plot(exc1_dist$EX.T1, exc1_dist$Dist, main="Tendon 1 Excursion VS Error Distance", xlab = "Tendon Excursion (mm)", ylab = "Error in mm")
exc2_dist <- select(exc2, 'EX.T2', 'Dist')
#print(exc2_dist)
plot(exc2_dist$EX.T2, exc2_dist$Dist, main="Tendon 2 Excursion VS Error Distance", xlab = "Tendon Excursion (mm)", ylab = "Error in mm")
exc3_dist <- select(exc3, 'EX.T3', 'Dist')
#print(exc3_dist)
plot(exc3_dist$EX.T3, exc3_dist$Dist, main="Tendon 3 Excursion VS Error Distance", xlab = "Tendon Excursion (mm)", ylab = "Error in mm")
exc4_dist <- select(exc4, 'EX.T4', 'Dist')
#print(exc4_dist)
plot(exc4_dist$EX.T4, exc4_dist$Dist, main="Tendon 4 Excursion VS Error Distance", xlab = "Tendon Excursion (mm)", ylab = "Error in mm")

#Based on dX
exc1_dist <- select(exc1, 'EX.T1', 'Dif.X')
#print(exc1_dist)
plot(exc1_dist$EX.T1, exc1_dist$Dif.X, main="Tendon 1 Excursion VS Error In X Value", xlab = "Tendon Excursion (mm)", ylab = "Error in mm")
exc2_dist <- select(exc2, 'EX.T2', 'Dif.X')
#print(exc2_dist)
plot(exc2_dist$EX.T2, exc2_dist$Dif.X, main="Tendon 2 Excursion VS Error In X Value", xlab = "Tendon Excursion (mm)", ylab = "Error in mm")
exc3_dist <- select(exc3, 'EX.T3', 'Dif.X')
#print(exc3_dist)
plot(exc3_dist$EX.T3, exc3_dist$Dif.X, main="Tendon 3 Excursion VS Error In X Value", xlab = "Tendon Excursion (mm)", ylab = "Error in mm")
exc4_dist <- select(exc4, 'EX.T4', 'Dif.X')
#print(exc4_dist)
plot(exc4_dist$EX.T4, exc4_dist$Dif.X, main="Tendon 4 Excursion VS Error In X Value", xlab = "Tendon Excursion (mm)", ylab = "Error in mm")

#Based on DY
exc1_dist <- select(exc1, 'EX.T1', 'Dif.Y')
#print(exc1_dist)
plot(exc1_dist$EX.T1, exc1_dist$Dif.Y, main="Tendon 1 Excursion VS Error In Y Value", xlab = "Tendon Excursion (mm)", ylab = "Error in mm")
exc2_dist <- select(exc2, 'EX.T2', 'Dif.Y')
#print(exc2_dist)
plot(exc2_dist$EX.T2, exc2_dist$Dif.Y, main="Tendon 2 Excursion VS Error In Y Value", xlab = "Tendon Excursion (mm)", ylab = "Error in mm")
exc3_dist <- select(exc3, 'EX.T3', 'Dif.Y')
#print(exc3_dist)
plot(exc3_dist$EX.T3, exc3_dist$Dif.Y, main="Tendon 3 Excursion VS Error In Y Value", xlab = "Tendon Excursion (mm)", ylab = "Error in mm")
exc4_dist <- select(exc4, 'EX.T4', 'Dif.Y')
#print(exc4_dist)
plot(exc4_dist$EX.T4, exc4_dist$Dif.Y, main="Tendon 4 Excursion VS Error In Y Value", xlab = "Tendon Excursion (mm)", ylab = "Error in mm")

dev.off()
