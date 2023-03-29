######PACCHETTI#######
library(lubridate)
library(dplyr)
library(xts)
library(forecast)
library(KFAS)
library(tsfknn)
library(Rcpp)
library(tseries)
library(tidymodels)
library(tidyverse)
library(timetk)
library(splusTimeDate)


#########IMPORT DATASET#######

df <- read.csv('data2022_train.csv')

#trasformazione in oggetto xts
dfts <- xts(df[, -1],
            ymd_hms(df$X)) 


#########VISUALIZZAZIONE SERIE###########
par(mfrow = c(3, 1))
plot(dfts, main='Serie completa') 
plot(dfts[43777:48096], main='Ultimo mese (novembre)') 
plot(dfts[47089:48096], main='Ultima settimana')
par(mfrow = c(1, 1))

#####CONTROLLO BOX-COX PER TRASFORMAZIONE LOGARITMICA#############

BoxCox.lambda(dfts) #0.2212144, valore intermedio che dice poco

adf.test(diff(dfts,144)) #p value 0.01 --> posso rifiutare ipotesi nulla di non stazionarietà

######CREAZIONE DUMMY CON GIORNI DELLA SETTIMANA PER STAGIONALITA SETTIMANALE####
df_dummy <- df %>% 
  mutate(
    days = lubridate::wday(X), #wday
    dummy = 1)

df_dummy <- tidyr::pivot_wider(data = df_dummy, names_from = days, values_from = dummy, values_fill = list(dummy = 0))

df_dummy <- as.matrix(df_dummy[,c(3:5, 7:9)]) #tolgo dummy del mercoledì dalla tabella

colnames(df_dummy) <- paste0('weekday', c(1,2,3,5,6,7)) #notazione anglosassone, giorno 1=domenica


#oggetto xts con sia la serie che le dummy
dfts_dum <- cbind(dfts, df_dummy) 

####CREAZIONE SINUSOIDI PER STAGIONALITA GIORNALIERA#######
omega <- outer(1:nrow(dfts), 1:10)*2*pi/144 #da 1 al numero di righe, 16 sinusoidi

cc <- cos(omega)
ss <- sin(omega)

colnames(cc) <- paste0('cos', c(1:10))
colnames(ss) <- paste0('sin', c(1:10))

#plot di controllo della frequenza
#plot(cc[1:1008, 1], type='l')

dfts_dum_sin <- cbind(dfts_dum, cc)
dfts_dum_sin <- cbind(dfts_dum_sin, ss)


#######DIVISIONE TRAIN-VALIDATION#########

train <- window(dfts_dum_sin, end = "2017-10-31 23:50:00")
val <- window(dfts_dum_sin, start = "2017-11-01 00:00:00")

plot(dfts, main = 'Divisione train-validation')
addEventLines(events = xts(x = '', order.by = as.Date('2017-11-01 00:00:00')),
              lty = 2, col = 'red', lwd = 2.5)


#creo anche i regressori divisi in train e test
reg_train <- as.matrix(train[, c(-1)])
reg_val <- as.matrix(val[, c(-1)])

reg_train_only_dum <- reg_train[, -c(7:26)]
reg_val_only_dum <- reg_val[, -c(7:26)]


#####REGRESSIONE LINEARE########
#primo tentativo semplice di regressione lineare con
#trend quadtratico e sinusoidi

t <- 1:nrow(dfts)
t2 <- t^2 #trend quadratico

#regressione lineare
reg1 <- lm(dfts ~ t + t2 + df_dummy + cc + ss)
summary(reg1)

#somma di componenti intercetta e trend --> livello
level <- reg1$coefficients[1] + 
  reg1$coefficients[2]*t + 
  reg1$coefficients[3]*t2

#stagionalit? annua costruita da sinusoidi
seas <- cc %*% reg1$coefficients[10:19] +
  ss %*% reg1$coefficients[20:29]

#grafico con stagionalità e livello
par(mfrow = c(3, 1))
plot(seas[1:1008], type='l', main = 'stagionalità', ylab='', xlab='') #stagionalità giornaliera
plot(level, type='l', main = 'livello', ylab='', xlab='') #livello
plot(df[, 'y'][1:1008], type='l', ylab='', xlab='', main='Stagionalità + livello vs serie reale')
lines(level0[1:1008] + seas[1:1008], col='red')
par(mfrow = c(1, 1))

#insieme delle componenti
plot(ts(seas)+level)

#settimana reale vs livello e stagionalità
level0 <- level - mean(level) + mean(df[, 'y']) #prendo livello sottraggo la sua media e sommo quella della serie storica
plot(df[, 'y'][1:1008], type='l', ylab='', xlab='',
     main='Stagionalità + livello vs serie reale', lwd=1.5)
lines(level0[1:1008] + seas[1:1008], col='red', lwd=1.5)

#acf e pacf residui
par(mfrow = c(2, 2))
Acf(reg1$residuals, 4032, main='ACF, 4032', ylab='')
Pacf(reg1$residuals, 4032, main='PACF, 4032', ylab='')

Acf(reg1$residuals, 1008)
Pacf(reg1$residuals, 1008)

Acf(reg1$residuals, main='ACF', ylab='')
Pacf(reg1$residuals, main='PACF', ylab='')
par(mfrow = c(1, 1))

#residui
plot(ts(reg1$residuals))



##########TEST MODELLI CON DIVERSE CONFIGURAZIONI########

#######ARIMA 1#################

arima1 <- Arima(y = train$dfts, #scelgo solo parte della serie come train set
              order = c(2, 0, 0),
              xreg = reg_train, 
              include.drift = T,
              method="CSS")
summary(arima1)

#controllo acf e pacf dei residui per vedere se sono normali
par(mfrow = c(1, 2))
Acf(arima1$residuals, 1008, main='ACF', ylab='')
Pacf(arima1$residuals, 1008, main='PACF', ylab='')
par(mfrow = c(1, 1))

plot(arima1$residuals, main='Residui', ylab='')

#previsioni e confronto con validation
prev_arima1 <- forecast(arima1,
                  nrow(val),
                  xreg = reg_val) #aggiungere xreg con tutti i regressori spostati in avanti 

plot(xts(prev_arima1$mean, time(val)), col='red', lwd=1, ylim = c(18000, 49000))
lines(val[, 1], col='black')

#calcolo mae per confronti
err_arima1 <- (as.vector(val$dfts) - as.vector(prev_arima1$mean)) #con start prendo solo 1 anno
mae_arima1 <- abs(err_arima1) %>% mean()
mae_arima1 #6530.932
mean(abs(err_arima1)/as.vector(val$dfts)*100) #23.35308


#######ARIMA 2#################

arima2 <- Arima(y = train$dfts, #scelgo solo parte della serie come train set
                order = c(3, 1, 0),
                #seasonal = list(order = c(0, 1, 0), period=144), #modello AIRLINE panico da provare
                xreg = reg_train, #anche qua uguale a sopra, scelgo solo parte della matrice dei regressori
                include.drift = T,
                method="CSS")
summary(arima2)

#controllo acf e pacf dei residui per vedere se sono normali
par(mfrow = c(1, 2))
Acf(arima2$residuals, 1008, main='ACF', ylab='')
Pacf(arima2$residuals, 1008, main='PACF', ylab='')
par(mfrow = c(1, 1))

plot(arima2$residuals, main='Residui', ylab='')

#previsioni e confronto con validation
prev_arima2 <- forecast(arima2,
                  nrow(val),
                  xreg = reg_val) #aggiungere xreg con tutti i regressori spostati in avanti 

plot(xts(prev_arima2$mean, time(val)), col='red', lwd=1, ylim = c(16000, 42000))
lines(val[, 1], col='black')

#calcolo mae per confronti
err_arima2 <- (as.vector(val$dfts) - as.vector(prev_arima2$mean)) #con start prendo solo 1 anno
mae_arima2 <- abs(err_arima2) %>% mean()
mae_arima2 #1901.971
mean(abs(err_arima2)/as.vector(val$dfts)*100) #6.456155


#######ARIMA 3#################

arima3 <- Arima(y = train$dfts, #scelgo solo parte della serie come train set
                order = c(3, 0, 0),
                seasonal = list(order = c(0, 1, 0), period=144), #modello AIRLINE panico da provare
                include.drift = T,
                method="CSS")
summary(arima3)

#controllo acf e pacf dei residui per vedere se sono normali
par(mfrow = c(1,2))
Acf(arima3$residuals, 1008, main='ACF', ylab='')
Pacf(arima3$residuals, 1008, main='PACF', ylab='')
par(mfrow = c(1, 1))

plot(arima3$residuals, main='Residui', ylab='')

#previsioni e confronto con validation
prev_arima3 <- forecast(arima3,
                  nrow(val)) #aggiungere xreg con tutti i regressori spostati in avanti 

plot(xts(prev_arima3$mean, time(val)), col='red', lwd=1, ylim = c(16000, 44000))
lines(val[, 1], col='black')

#calcolo mae per confronti
err_arima3 <- (as.vector(val$dfts) - as.vector(prev_arima3$mean)) #con start prendo solo 1 anno
mae_arima3 <- abs(err_arima3) %>% mean()
mae_arima3 #1858.162
mean(abs(err_arima3)/as.vector(val$dfts)*100) #6.692717


#######ARIMA 4#################

arima4 <- Arima(y = train$dfts, #scelgo solo parte della serie come train set
                order = c(3, 0, 0),
                seasonal = list(order = c(0, 1, 1), period=144), #modello AIRLINE panico da provare
                #xreg = reg_train, #anche qua uguale a sopra, scelgo solo parte della matrice dei regressori
                include.drift = T,
                method="CSS")
summary(arima4)

#controllo acf e pacf dei residui per vedere se sono normali
par(mfrow = c(1, 2))
Acf(arima4$residuals, 1008, main='ACF', ylab='')
Pacf(arima4$residuals, 1008, main='PACF', ylab='')
par(mfrow = c(1, 1))

plot(arima4$residuals, main='Residui', ylab='')

#previsioni e confronto con validation
prev_arima4 <- forecast(arima4,
                  nrow(val)) #aggiungere xreg con tutti i regressori spostati in avanti 

plot(xts(prev_arima4$mean, time(val)), col='#20a39e',
     lwd=1.5, ylim = c(18000,45000), main = 'Previsioni vs validation set')
lines(val[, 1], col='black')

#calcolo mae per confronti
err_arima4 <- (as.vector(val$dfts) - as.vector(prev_arima4$mean)) #con start prendo solo 1 anno
mae_arima4 <- abs(err_arima4) %>% mean()
mae_arima4 #1839.028

mean(abs(err_arima4)/as.vector(val$dfts)*100) #6.69136



##########UCM################

#preprocessing

df_ucm <- dfts_dum_sin
df_ucm$y <-df_ucm$dfts

df_ucm$y[(nrow(df_ucm)-4320):nrow(df_ucm)] <- NA

#calcolo varianza della serie intera
vy <- var(df$y)

####UCM1#######

ucm1 <- SSModel(y ~ 0 + #non voglio intercetta
                  SSMtrend(1, NA) + #random walk
                  SSMseasonal(144, 0, 'trigonometric', harmonics = 1:10), #altra stagionalità deterministica (0) e trigonometrica specificando quante armoniche
                H = NA, #errore di osservazione
                data = df_ucm #dicendogli data = dataset posso utilizzare i nomi delle variabili
)

#assegno nuovi valori alle matrici
diag(ucm1$P1inf) <- 0
diag(ucm1$P1) <- vy #metto varianza della serie storica
ucm1$a1['level', ] <- mean(train$dfts)

#distribuisco variazione della serie
pars <- log(c(
  logVarEta  = vy/10,
  logVarZeta = vy/1000,
  logVarOm7  = vy/1000,
  logVarOm365= vy/10000,
  logVarEps  = vy/10
))

ucm_fit1 <- fitSSM(ucm1, pars) #, updt1

#controllo che si arrivi a convergenza
ucm_fit1$optim.out$convergence 

#filtro di Kalman
smo1 <- KFS(ucm_fit1$model,
            filtering = 'signal',
            smoothing = c('state', 'disturbance'))

plot(ts(df_ucm$y))
lines(ts(smo1$alphahat[, 'level']), col = 'red')

#plot livello sulla serie
p_ucm1 <- smo1$m[(nrow(df_ucm)-4320+1):nrow(df_ucm)]
actual_ucm <- as.vector(df_ucm$dfts[(nrow(df_ucm)-4320+1):nrow(df_ucm)])

#plot serie totale e previsioni
plot(ts(df_ucm$dfts))
lines((nrow(df_ucm)-4320):nrow(df_ucm), smo1$m[(nrow(df_ucm)-4320):nrow(df_ucm)], col='red')

#plot serie e previsioni solo ultimo mese
plot(xts(smo1$m[(nrow(df_ucm)-4320+1):nrow(df_ucm)], time(val)), col='red', lwd=1, ylim = c(18000,43000))
lines(val[, 1], col='black')

#calcolo mae per confronti
err_ucm1 <- (actual_ucm - (smo1$m[(nrow(df_ucm)-4320+1):nrow(df_ucm)])) #con start prendo solo 1 anno
mae_ucm1 <- abs(err_ucm1) %>% mean()
mae_ucm1 #1990.408
mean(abs(err_ucm1)/as.vector(val$dfts)*100) #6.818897


####UCM2#######

ucm2 <- SSModel(y ~ 0 + #non voglio intercetta
                  SSMtrend(1, NA) + #random walk
                  SSMseasonal(144, 0, 'trigonometric', harmonics = 1:10) +
                  SSMseasonal(1008, 0, 'trigonometric', harmonics = 1:10), #altra stagionalità deterministica (0) e trigonometrica specificando quante armoniche
                H = NA, #errore di osservazione
                data = df_ucm #dicendogli data = dataset posso utilizzare i nomi delle variabili
)

#assegno nuovi valori alle matrici
diag(ucm2$P1inf) <- 0
diag(ucm2$P1) <- vy #metto varianza della serie storica
ucm2$a1['level', ] <- mean(train$dfts)

ucm_fit2 <- fitSSM(ucm2, ucm_fit1$optim.out$par) #, updt1

#controllo che si arrivi a convergenza
ucm_fit2$optim.out$convergence 

#filtro di Kalman
smo2 <- KFS(ucm_fit2$model,
            filtering = 'signal',
            smoothing = c('state', 'disturbance'))

plot(ts(df_ucm$y))
lines(ts(smo2$alphahat[, 'level']), col = 'red')

#plot serie totale e previsioni
plot(ts(df_ucm$dfts))
lines((nrow(df_ucm)-4320):nrow(df_ucm), smo2$m[(nrow(df_ucm)-4320):nrow(df_ucm)], col='red')

#plot serie e previsioni solo ultimo mese
plot(xts(smo2$m[(nrow(df_ucm)-4320+1):nrow(df_ucm)], time(val)), col='#ef5b5b',
     lwd=1.5, ylim = c(15000, 44000), main='Previsioni vs validation set')
lines(val[, 1], col='black')

#calcolo mae per confronti
err_ucm2 <- (actual_ucm - (smo2$m[(nrow(df_ucm)-4320+1):nrow(df_ucm)])) #con start prendo solo 1 anno
mae_ucm2 <- abs(err_ucm2) %>% mean()
mae_ucm2 #1879.281

mean(abs(err_ucm2)/as.vector(val$dfts)*100) #6.475925


####UCM3#######

ucm3 <- SSModel(y ~ 0 + #non voglio intercetta
                  SSMtrend(1, NA) + #random walk
                  SSMseasonal(144, 0, 'trigonometric', harmonics = 1:10) +
                  SSMregression(dfts ~ weekday1 + weekday2 +
                                  weekday3 + weekday5 +
                                  weekday6 + weekday7,
                                data=df_ucm), #altra stagionalità deterministica (0) e trigonometrica specificando quante armoniche
                H = NA, #errore di osservazione
                data = df_ucm #dicendogli data = dataset posso utilizzare i nomi delle variabili
)

#assegno nuovi valori alle matrici
diag(ucm3$P1inf) <- 0
diag(ucm3$P1) <- vy #metto varianza della serie storica
ucm3$a1['level', ] <- mean(train$dfts)

ucm_fit3 <- fitSSM(ucm3, ucm_fit2$optim.out$par) #, updt1

#controllo che si arrivi a convergenza
ucm_fit3$optim.out$convergence 

#filtro di Kalman
smo3 <- KFS(ucm_fit3$model,
            filtering = 'signal',
            smoothing = c('state', 'disturbance'))

plot(ts(df_ucm$y))
lines(ts(smo3$alphahat[, 'level']), col = 'red')

#plot serie totale e previsioni
plot(ts(df_ucm$dfts))
lines((nrow(df_ucm)-4320):nrow(df_ucm), smo3$m[(nrow(df_ucm)-4320):nrow(df_ucm)], col='red')

#plot serie e previsioni solo ultimo mese
plot(xts(smo3$m[(nrow(df_ucm)-4320+1):nrow(df_ucm)], time(val)), col='red', lwd=1, ylim = c(18000,43000))
lines(val[, 1], col='black')

#calcolo mae per confronti
err_ucm3 <- (actual_ucm - (smo3$m[(nrow(df_ucm)-4320+1):nrow(df_ucm)])) #con start prendo solo 1 anno
mae_ucm3 <- abs(err_ucm3) %>% mean()
mae_ucm3 #1983.751

mean(abs(err_ucm3)/as.vector(val$dfts)*100) #6.794245



#######KNN############

#preprocessing
y <- xts(df[, -1],
         ymd_hms(df$X))

train_ndx <- 1:(nrow(y)-4320) #train primi 10 mesi
test_ndx <- (nrow(y)-4320+1):nrow(y) #test mese novembre


##########KNN1###########

kn1 <- knn_forecasting(as.vector(train$dfts),
                       h = nrow(val),
                       lags = 1:1008, #non so se abbia senso mettere 1008, forse meglio 4320 come quello che devo prevedere
                       k = 10,
                       msas = "MIMO",
                       cf = "median",
                       transform = "none")

#mae, rmse e mape
mae_kn1 <- mean(abs(as.vector(val$dfts) - kn1$prediction))
mae_kn1 #2346.035
err_kn1 <- (val$dfts - kn1$prediction)
mean(abs(err_kn1)/as.vector(val$dfts)*100) #mape 8.034549

#plot tutto il mese di novembre con previsioni
plot(xts(kn1$prediction, time(val)), col='#ffba49',
     lwd=1, ylim = c(18000,47000), main = 'Previsioni vs validation set')
lines(val[, 1], col='black')

plot(ts(df_ucm$dfts))
lines(kn1$prediction, col='red')



##########KNN2###########

kn2 <- knn_forecasting(as.vector(train$dfts),
                       h = nrow(val),
                       lags = 1:4320, #non so se abbia senso mettere 1008, forse meglio 4320 come quello che devo prevedere
                       k = 10,
                       msas = "MIMO",
                       cf = "median",
                       transform = "none")

#mae, rmse e mape
mae_kn2 <- mean(abs(as.vector(val$dfts) - kn2$prediction))
mae_kn2 # 3065.367
err_kn2 <- (val$dfts - kn2$prediction)
mean(abs(err_kn2)/as.vector(val$dfts)*100) #mape 10.82601

#plot tutto il mese di novembre con previsioni
plot(xts(kn2$prediction, time(val)), col='red', lwd=1, ylim = c(18000,47000))
lines(val[, 1], col='black')

plot(ts(df_ucm$dfts))
lines(kn2$prediction, col='red')


##########KNN3###########

kn3 <- knn_forecasting(as.vector(train$dfts),
                       h = nrow(val),
                       lags = 1:1008, #non so se abbia senso mettere 1008, forse meglio 4320 come quello che devo prevedere
                       k = 5,
                       msas = "MIMO",
                       cf = "median",
                       transform = "none")

#mae, rmse e mape
mae_kn3 <- mean(abs(as.vector(val$dfts) - kn3$prediction))
mae_kn3
err_kn3 <- (val$dfts - kn3$prediction)
mean(abs(err_kn3)/as.vector(val$dfts)*100) #mape

#plot tutto il mese di novembre con previsioni
plot(xts(kn3$prediction, time(val)), col='red', lwd=1, ylim = c(18000,47000))
lines(val[, 1], col='black')

plot(ts(df_ucm$dfts))
lines(kn3$prediction, col='red')


##########KNN4###########

kn4 <- knn_forecasting(as.vector(train$dfts),
                       h = nrow(val),
                       lags = 1:1008, #non so se abbia senso mettere 1008, forse meglio 4320 come quello che devo prevedere
                       k = 20,
                       msas = "MIMO",
                       cf = "median",
                       transform = "none")

#mae, rmse e mape
mae_kn4 <- mean(abs(as.vector(val$dfts) - kn4$prediction))
mae_kn4
err_kn4 <- (val$dfts - kn4$prediction)
mean(abs(err_kn4)/as.vector(val$dfts)*100) #mape

#plot tutto il mese di novembre con previsioni
plot(xts(kn4$prediction, time(val)), col='red', lwd=1, ylim = c(18000,47000))
lines(val[, 1], col='black')

plot(ts(df_ucm$dfts))
lines(kn4$prediction, col='red')


##########################################
#per avere previsioni su dati nuovi non visti devo riallenare il modello senza dividere train e test
#prevede 4464 valori, tutto dicembre
kn2 <- knn_forecasting(as.vector(dfts[,1]),
                       h = 4464,
                       lags = 1:1008, #non so se abbia senso mettere 1008, forse meglio 4320 come quello che devo prevedere
                       k = 10,
                       msas = "MIMO",
                       cf = "median",
                       transform = "none")

plot(as.numeric(y), type = "l", xlim=c(0, 52560)) #, ylim=c(18000, 45000)
lines(kn2$prediction, col = "red")

plot(dfts, lwd=1, end.time='2017-12-31 23:50:00')
lines(dfts[1:20000, ], col='red')


dfts_total <- df

dfts_total[(nrow(dfts_total)+4464),] <- NA

dfts_total <- xts(dfts_total[, -1],
            ymd_hms(dfts_total$X))

plot(ts(dfts_total$y), xlim=c(0, 52560))
lines(kn2$prediction, col='red')




############MIGLIORI MODELLI PER PREVISIONI#############

#########BEST ARIMA#########

arima_best <- Arima(y = dfts, #scelgo solo parte della serie come train set
                order = c(3, 0, 0),
                seasonal = list(order = c(0, 1, 1), period=144), #modello AIRLINE panico da provare
                #xreg = reg_train, #anche qua uguale a sopra, scelgo solo parte della matrice dei regressori
                include.drift = T,
                method="CSS")

#previsioni e confronto con validation
final_prev_arima <- forecast(arima_best,
                  4464) #aggiungere xreg con tutti i regressori spostati in avanti 

p_arima <- final_prev_arima$mean


########BEST UCM################
val_list <- df[,2]
for (i in c(1:4464)){
  val_list <- append(val_list, NA) 
}
start <- as.POSIXct("2017-01-01 00:00:00")
t <- seq(from = start, length.out = 52560, by = "10 mins")
df_ucm_final <- as.data.frame(cbind(t, val_list))

vy <- var(df$y)

df_ucm <- dfts_dum_sin
df_ucm$y <-df_ucm$dfts

df_ucm$y[(nrow(df_ucm)-4320):nrow(df_ucm)] <- NA

ucm1 <- SSModel(y ~ 0 + #non voglio intercetta
                  SSMtrend(1, NA) + #random walk
                  SSMseasonal(144, 0, 'trigonometric', harmonics = 1:10), #altra stagionalità deterministica (0) e trigonometrica specificando quante armoniche
                H = NA, #errore di osservazione
                data = df_ucm #dicendogli data = dataset posso utilizzare i nomi delle variabili
)

diag(ucm1$P1inf) <- 0
diag(ucm1$P1) <- vy #metto varianza della serie storica
ucm1$a1['level', ] <- mean(train$dfts)

pars <- log(c(
  logVarEta  = vy/10,
  logVarZeta = vy/1000,
  logVarOm7  = vy/1000,
  logVarOm365= vy/10000,
  logVarEps  = vy/10
))

ucm_fit1 <- fitSSM(ucm1, pars)

ucm_best <- SSModel(val_list ~ 0 + #non voglio intercetta
                  SSMtrend(1, NA) + #random walk
                  SSMseasonal(144, 0, 'trigonometric', harmonics = 1:10) +
                  SSMseasonal(1008, 0, 'trigonometric', harmonics = 1:10), #altra stagionalità deterministica (0) e trigonometrica specificando quante armoniche
                H = NA, #errore di osservazione
                data = df_ucm_final #dicendogli data = dataset posso utilizzare i nomi delle variabili
)

#assegno nuovi valori alle matrici
diag(ucm_best$P1inf) <- 0
diag(ucm_best$P1) <- vy #metto varianza della serie storica
ucm_best$a1['level', ] <- mean(train$dfts)

ucm_fit_best <- fitSSM(ucm_best, ucm_fit1$optim.out$par) #, updt1

#controllo che si arrivi a convergenza
ucm_fit_best$optim.out$convergence 

#filtro di Kalman
smo_best <- KFS(ucm_fit_best$model,
            filtering = 'signal',
            smoothing = c('state', 'disturbance'))

#plot serie totale e previsioni
#plot(ts(df_ucm_final$val_list))
#lines((nrow(df_ucm_final)-4464):nrow(df_ucm_final), smo_best$m[(nrow(df_ucm_final)-4464):nrow(df_ucm_final)], col='red')

#plot serie e previsioni solo ultimo mese
#plot(xts(smo_best$m[(nrow(df_ucm)-4464+1):nrow(df_ucm)], time(val)), col='red', lwd=1, ylim = c(18000,43000))
#lines(val[, 1], col='black')

p_ucm_best <- smo_best$m[(nrow(df_ucm_final)-4464+1):nrow(df_ucm_final)]


#########BEST KNN###############

kn_best <- knn_forecasting(as.vector(dfts[,1]),
                       h = 4464,
                       lags = 1:1008, #non so se abbia senso mettere 1008, forse meglio 4320 come quello che devo prevedere
                       k = 10,
                       msas = "MIMO",
                       cf = "median",
                       transform = "none")

p_kn <- kn_best$prediction


#########UNIONE PREVISIONI PER VISUALIZZAZIONE########

#creazione colonna con date per file previsioni
now <- as.POSIXct("2017-12-01 00:00:00")
tseq <- seq(from = now, length.out = 4464, by = "10 mins")

#creazione file data con tutte le previsioni unite
data <- data.frame(tseq, p_arima, p_kn, p_ucm_best)
colnames(data) <- c('date', 'ARIMA', 'UCM', 'ML')
write.csv(data, '828725_20230113.csv', row.names = FALSE)

#trasformazione in xts per visualizzazione tutte assieme
data <- xts(data[, -1], ymd_hms(data$date))
plot.xts(data, col=c('#20a39e', '#ef5b5b', '#ffba49'),
         lwd=1.5,
         multi.panel = TRUE,
         main='Forecast dicembre 3 metodi')

plot.xts(data, col=c('#20a39e', '#ef5b5b', '#ffba49'),
         lwd=1.5, legend.loc = 'bottomright',
         main ='Forecast dicembre comparazione metodi')


#visualizzazioni serie complete + prevesioni

tot_year <- seq(from = as.POSIXct("2017-01-01 00:00:00"), length.out = 52560, by = "10 mins")

#arima
tot_arima <- c(df$y, as.vector(p_arima))
tot_arima <- data.frame(tot_year, tot_arima)
tot_arima <- xts(tot_arima[, -1], ymd_hms(tot_arima$tot_year))
plot.xts(tot_arima, col='#20a39e',
         lwd=1.5, main ='ARIMA')

#ucm
tot_ucm_best <- c(df$y, as.vector(p_ucm_best))
tot_ucm_best <- data.frame(tot_year, tot_ucm_best)
tot_ucm_best <- xts(tot_ucm_best[, -1], ymd_hms(tot_ucm_best$tot_year))
plot.xts(tot_ucm_best, col='#ef5b5b',
         lwd=1.5, main ='UCM')

#ml
tot_kn <- c(df$y, as.vector(p_kn))
tot_kn <- data.frame(tot_year, tot_kn)
tot_kn <- xts(tot_kn[, -1], ymd_hms(tot_kn$tot_year))
plot.xts(tot_kn, col='#ffba49',
         lwd=1.5, main ='KNN')


#previsioni con lstm, da analizzare dopo per far vedere che non è stato scelto
p_lstm <- read.csv('LSTM_pred.csv')
p_lstm <- p_lstm[,2]

prev_ml <- data.frame(tseq, p_lstm, p_kn)
colnames(prev_ml)<- c('date', 'LSTM', 'KNN')
prev_ml <- xts(prev_ml[, -1], ymd_hms(prev_ml$date))
plot.xts(prev_ml, col=c('black', '#ffba49'),
         lwd=1.5, legend.loc = 'bottomright',
         main='Comparazione metodi ML')

#comparazione varianze per motivazioni
var(df$y) #varianza serie intera    50906683
var(p_lstm) #varianza lstm          837826
var(p_kn) #varianza knn             45765264

#########################















prev_tot <- read.csv('PREV_FINAL.csv')
prev_tot$X <- tseq

pp_arima <- prev_tot[, c(1,2)]
colnames(pp_arima) <- c('X', 'y')
pp_kn <- prev_tot[, c(1,3)]
colnames(pp_kn) <- c('X', 'y')
pp_ucm <- prev_tot[, c(1,4)]
colnames(pp_ucm) <- c('X', 'y')

df <- within(df, X <- as.POSIXct(as.character(X), format = "%Y-%m-%d %H:%M:%S"))
pp_arima <- within(pp_arima, X <- as.POSIXct(as.character(X), format = "%Y-%m-%d %H:%M:%S"))
pp_kn <- within(pp_kn, X <- as.POSIXct(as.character(X), format = "%Y-%m-%d %H:%M:%S"))
pp_ucm <- within(pp_ucm, X <- as.POSIXct(as.character(X), format = "%Y-%m-%d %H:%M:%S"))


z_arima <- rbind(df, pp_arima)
z_ucm <- rbind(df, pp_ucm)
z_kn <- rbind(df, pp_kn)


autoplot(ts(z_arima$y), alpha=0.5, col='purple')
lines(ts(z_ucm$y), alpha=0.5, col='green')
autoplot(ts(z_kn$y), alpha=0.5, col='blue')

#prova da k-medie
stream <- df$y
plot(c(df$y, numeric(4464)+NA), type='l', lwd=1)
lines((48096+1):(48096+4464), z_arima$y[(48096+1):(48096+4464)],
      lwd=1, alpha = 0.3, col=rgb(red = 0, green = 0, blue = 0.5, alpha = 0.3))
lines((48096+1):(48096+4464), z_ucm$y[(48096+1):(48096+4464)],
      lwd=1, alpha = 0.3, col=rgb(red = 0, green = 0.5, blue = 0, alpha = 0.3))
lines((48096+1):(48096+4464), z_kn$y[(48096+1):(48096+4464)],
      lwd=1, alpha = 0.3, col=rgb(red = 0.5, green = 0, blue = 0, alpha = 0.3))





z_arima
z_arima <- xts(z_arima[, -1], ymd_hms(z_arima$X))
plot(z_arima)

tot_arima <- cbind(t, c(df$y, p_arima))
tot_arima[,1] <- t
##############################





























colnames(data) <- c('X', 'y')
data$X <- ymd_hms(data$X)

t <- rbind(dfts, data)
t<- as.data.frame(t)

plot(t[,1], col = ifelse(t < '2017-11-30 23:50:00', "blue", "red"), type='l')

plot(xts(t[, -1], ymd_hms(t$X)))




dat <- data.frame(x = 1:10, y = rep(c("a", "b"), each = 5))

library(lattice)
xyplot(x ~ 1:length(x), data = dat, group = y, type = "b", 
       col = c("black", "red"))




df$X <- ymd_hms(df$X)
df$y <- as.numeric(df$y)

plot_time_series(df, X, y, .interactive=FALSE, .smooth = FALSE)



