########## Prevendo a Inadimplência de Clientes com Machine Learning ##########
#																			  #
#																			  #
#																			  #
#																		      #																			
###############################################################################

# Instalando os pacotes para o projeto
install.packges("Amelia")
install.packges("caret")
install.packges("ggplot2")
install.packges("dplyr")
install.packges("reshape")
install.packges("randomForest")
install.packges("e1071")

# Carregando os pacotes
library(Amelia)
library(ggplot2)
library(caret)
library(reshape)
library(randomForest)
library(dplyr)
library(e1071)

# Carregando o dataset
# Fonte: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
dados_clientes <- read.csv("dados/dataset.csv")

# Visualizando os dados e sua estrutura
View(dados_clientes)
dim(dados_clientes)
str(dados_clientes)
summary(dados_clientes)

############### Análise Exploratória, Limpeza e Transformação ##################

# Removendo a primeira coluna ID
dados_clientes$ID <-NULL
dim(dados_clientes)
View(dados_clientes)

# Renomeando a coluna de classe
colnames(dados_clientes)
colnames(dados_clientes) [24] <- "Inadimplente"
colnames(dados_clientes)
View(dados_clientes)

# Verificando valores ausentes e removendo do dataset
sapply(dados_clientes, function(x) sum(is.na(x)))
?missmap
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)

# Convertendo os atributos gênero, escolaridade, estado civil e idadade para fatores (categorias)

# Renomeando colunas categóricas
colnames(dados_clientes)
colnames(dados_clientes) [2] <- "Genero"
colnames(dados_clientes) [3] <- "Escolaridade"
colnames(dados_clientes) [4] <- "Estado_Civil"
colnames(dados_clientes) [5] <- "Idade"
colnames(dados_clientes)

View(dados_clientes)

# Gênero
View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)
?cut

dados_clientes$Genero <- cut(dados_clientes$Genero,
	                         c(0,1,2),
	                         labels = c("Masculino",
	                         	        "Feminino"))

View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)

# Escolaridade
View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)
dados_clientes$Escolaridade <- cut(dados_clientes$Escolaridade,
	                               c(0,1,2,3,4),
	                               labels = c("Pos Graduado",
	                               	          "Graduado",
	                               	          "Ensino Medio",
	                               	          "Outros"))

View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)

# Estado Civil
View(dados_clientes$Estado_Civil)
str(dados_clientes$Estado_Civil)
summary(dados_clientes$Estado_Civil)
dados_clientes$Estado_Civil <- cut(dados_clientes$Estado_Civil,
	                               c(-1,0,1,2,3),
	                               labels = c("Desconhecido",
	                               	          "Casado",
	                               	          "Solteiro",
	                               	          "Outro"))

View(dados_clientes$Estado_Civil)
str(dados_clientes$Estado_Civil)
summary(dados_clientes$Estado_Civil)

# Convertendo a variável para o tipo fator com faxia etária
str(dados_clientes$Idade)
summary(dados_clientes$Idade)
hist(dados_clientes$Idade)
dados_clientes$Idade <- cut(dados_clientes$Idade,
	                        c(0,30,50,100),
	                        labels = c("Jovem",
	                        	       "Adulto",
	                        	       "Idoso"))

View(dados_clientes$Idade)
str(dados_clientes$Idade)
summary(dados_clientes$Idade)

# Convertendo a variável que indica pagamentos para o tipo fator
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

# Dataset após a conversões
str(dados_clientes)
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)
missmap(dados_clientes, main = "Valores Missing Observados")
dim(dados_clientes)


#######################################################################


# Alterando a variável dependente para o tipo fator
str(dados_clientes$Inadimplente)
colnames(dados_clientes)
dados_clientes$Inadimplente <- as.factor(dados_clientes$Inadimplente)
str(dados_clientes$Inadimplente)
View(dados_clientes)

# Total de Inadimplentes versus não-inadimplentes
table(dados_clientes$Inadimplente)

# Verificando as porcentagens entre as classes
prop.table(table(dados_clientes$Inadimplente))

# Plot da distribuição usando o ggplot2
qplot(Inadimplente, data = dados_clientes, geom = "bar") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Set seed
set.seed(12345)

# Amostragem Estratificada
# Seleciona as linhas de acordo com a variável inadimplente como strata
?createDataPartition
indice <- createDataPartition(dados_clientes$Inadimplente, p = 0.75, list = FALSE)
dim(indice)

# Definimos os dados de treinamento como subconjunto do conjunto de dados original
# Com números de índice de linha (conforme identificado acima) e todas as colunas
dados_treino <- dados_clientes[indice,]
dim(dados_treino)
table(dados_treino$Inadimplente)

# Veja porcentagens entre as classes
prop.table(table(dados$Inadimplente))

# Número de registros no dataset de treinamento
dim(dados_treino)

# Comparamos as porcentagens entre as classes de treinamento e dados originais
compara_dados <- cbind(prop.table(table(dados_treino$Inadimplente)),
					   prop.table(table(dados_clientes$Inadimplente)))
colnames(compara_dados) <- c("Treinamento", "Original")
compara_dados

# Melt Data - Converte colunas em linhas
?reshape2::melt
melt_compara_dados <- melt(compara_dados)
melt_compara_dados

# Plot para visualizar a distribuição do treinamento vs original
ggplot(melt_compara_dados, aes(x = X1, y = value))+
	geom_bar(aes(fill = X2), stat = "identity", position = "dodge")+
	theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Tudo o que não está no dataset de treinamento está no dataset de teste. Observe o sinal - (menos)
dados_teste <- dados_clientes[-indice,]
dim(dados_teste)
dim(dados_treino)


#####################################################################

#################### Modelo de Machine Learning #####################

# Constuindo a primeira versão do modelo
?randomForest
modelo_v1 <- randomForest(Inadimplente ~ ., data = dados_treino)
modelo_v1

# Avaliando o modelo
plot(modelo_v1)

# Previsões com dados de teste
previsoes_v1 <- predict(modelo_v1, dados_teste)

# Confusion Matrix
?caret::confusionMatrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$Inadimplente, positive = "1")
cm_v1

# Calculando Precision, Recall e F1-Score, métricas de avaliação do modelo preditivo
y <- dados_teste$Inadimplente
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision

recall <- sensitivity(y_pred_v1, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)