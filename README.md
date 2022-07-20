# Processamento de dados em Larga Escala - Corpus PT7 multiclasse

## Dupla

- Mariama Celi Serafim de Oliveira - mcso

- Maria Tassiane Barros de Lima - mtbl

# Etapas do projeto
## ETL

O [PT7 Web](https://ieee-dataport.org/open-access/pt7-web-annotated-portuguese-language-corpus) é um Corpus anotado em língua portuguesa construído a partir de amostras coletadas de setembro de 2018 a março de 2020 de sete países de língua portuguesa: Angola, Brasil, Portugal, Cabo Verde, Guiné-Bissau, Macau e Moçambique.

A partir do Corpus PT7, foi feita uma tarefa de classificação multiclasse, com o objetivo de identificar de país de língua portuguesa o texto extraído pertence. Para tal, foram utilizados dois scripts ([```labels-pt7-raw.scala```](labels-pt7-raw.scala), [```etl-pt7.scala```](etl-pt7.scala)) no Spark que realizam a extração das features e do rótulo. 

## Treinamento

Após a etapa de ETL, temos um conjunto de 17014 observações com 6 diferentes classes, que servirá entrada para o classificador escolhido.

O classificador escolhido foi Random Forest com os seguintes hiperparâmetros, ```numTrees:10``` e ```maxDepth:10```, a implementação completa está no arquivo [```pt7_random_forest.scala```](pt7_random_forest.scala)<sup>1</sup>. 

Foi realizada uma divisão de 70% para treino e 30% teste, com uma SEED fixa de valor 42.

Além disso, foi utilizado a ferramenta de pipeline da biblioteca spark.ml. No pipeline, o primeiro passo foi a indexação de rótulos textuais para valores numéricos. Logo após, foi realizado o treinamento.

<sup>1</sup>OBS.: Como o projeto foi realizado para fins pedagógicos, não realizamos a etapa de encontrar os melhores hiperpâmetros, no entanto consideramos essa etapa fundamental para a obtenção do melhor modelo. A equipe ainda tentou realizar o Cross Validation, no entanto a abordagem demandou bastante poder computacional.

## Teste sobre Corpus PT7 multiclasse

As métricas utilizadas para avaliar a performance do modelo foram: Accuracy, False Positive Rate, FMeasure, Precision, Recall, True Positive Rate. Ademais, exibimos uma matriz de confusão. Todos os resultados podem ser encontrados no arquivo [```metrics.txt```](metrics.txt).

## Salvando o modelo

Ao final, o modelo gerado foi salvo no sistema HDFS do cluster. O arquivo pode ser encontrado no cluster a partir do seguinte caminho ```hdfs://master:8020/bigdata/model_random_forest```.

# Executando o projeto

1. Inicie a imagem do cluster fornecido para a disciplina.

2. **Etapa de ETL**
- Copie os arquivos do corpus PT7 Web, [```labels-pt7-raw.scala```](labels-pt7-raw.scala) e [```etl-pt7.scala```](etl-pt7.scala) para o diretório ```user_data```.
- Copie os arquivos do corpus para o HDFS. 
```
hadoop fs -put /user_data/pt7-raw hdfs://master:8020/bigdata/
```
- Por último, execute os arquivos ```labels-pt7-raw.scala``` e ```etl-pt7.scala``` em sequência, utilizando os comandos abaixo:
```
spark-shell --master spark://master:7077 -i /user_data/labels-pt7-raw.scala

spark-shell --master spark://master:7077 -i /user_data/etl-pt7.scala
```

3. **Treinando o classificador e coletando métricas**

- Copie o arquivo [```pt7_random_forest.scala```](pt7_random_forest.scala) na pasta ```user_data```.

- Inicialize o bash Spark:
```
docker exec -it master spark-shell --master spark://master:7077
```
- Dentro do bash Spark, utilize o comando abaixo para rodar script scala.
```
:load /user_data/pt7_random_forest.scala
```
- As métricas serão salvas no ```user_data``` no arquivo 
 ```metrics.txt```, e o modelo no HDFS no caminho  ```hdfs://master:8020/bigdata/model_random_forest```.
