/////////////////////////////////////////////////////////////
//Alunas:    Maria Tassiane Barros de Lima (mtbl),         //
//           Mariama Celi Serafim de Oliveira (mcso)       //
/////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////
//                       Bibliotecas                       //
/////////////////////////////////////////////////////////////
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import java.io.File
import java.io.PrintWriter

import spark.implicits._


/////////////////////////////////////////////////////////////
//             Carregando e Indexando Dados                //
/////////////////////////////////////////////////////////////

// Lendo os dados do HDFS e converte para um dataframe
val df = { 
	spark.read
	.format("parquet")
	.load("hdfs://master:8020/bigdata/pt7-hash.parquet")
}

//Indexando coluna de labels
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(df)
  
// Separando dataset em treino e teste (30% para teste e 70% para treino).
// val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), 42)

val Array(trainingData, testData) = df.randomSplit(Array(0.01, 0.3), 42)
//treino, teste = a.randomSplit(weights = [0.7, 0.3], seed = SEED)


/////////////////////////////////////////////////////////////
//             Instanciando modelo e treinando             //
/////////////////////////////////////////////////////////////

// Instanciando o classificador - Random Forest
val rf = new RandomForestClassifier()
		.setLabelCol("indexedLabel")
		.setFeaturesCol("features")
	  .setNumTrees(10)
    .setMaxDepth(10)

// Criando pipeline para o Random Forest
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, rf))

// Treinando o modelo
val model = pipeline.fit(trainingData)

// Realizando predições
val predictions = model.transform(testData)

// Selecionadno e exibindo parte das prediçoes
predictions.select("prediction","indexedLabel").show(5)

// Selecionando as colunas "prediction" e "indexLabel"
val preds_and_labels = predictions.select("prediction","indexedLabel")

// Calculando as métricas
//Instanciando objeto que calcula métricas 
val metrics_ = new MulticlassMetrics(preds_and_labels.rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))


/////////////////////////////////////////////////////////////
//               Salvando Métricas e Modelo                //
/////////////////////////////////////////////////////////////

// Extraindo labels
val labels = metrics_.labels

//Escrever métricas no arquivo
val file_object = new File("/user_data/metrics.txt" ) 
val print_writer = new PrintWriter(file_object) 

// Matrix de Confusão
val confusionMatrixOutput = metrics_.confusionMatrix.toString
print_writer.write(s"Matrix de Confusão:\n$confusionMatrixOutput")
print_writer.write("\n\n")

// Taxa de Falso Positivo por label
print_writer.write(s"Taxa de Falso Positivo:\n")
labels.foreach { l =>
  print_writer.write(s"FPR($l) = " + metrics_.falsePositiveRate(l) + "\n")
}
print_writer.write("\n")

// Taxa de Verdadeiro Positivo por label
print_writer.write(s"Taxa de Verdadeiro Positivo:\n")
labels.foreach { l =>
  print_writer.write(s"TPR($l) = " + metrics_.truePositiveRate(l) + "\n")
}
print_writer.write("\n")

// F1-Score por label
print_writer.write(s"F1-Score por label:\n")
labels.foreach { l =>
  print_writer.write(s"F1-Score($l) = " + metrics_.fMeasure(l) + "\n")
}
print_writer.write("\n")

// Precision por label
print_writer.write(s"Precision:\n")
labels.foreach { l =>
  print_writer.write(s"Precision($l) = " + metrics_.precision(l) + "\n")
}
print_writer.write("\n")

// Recall by label
print_writer.write(s"Recall por label:\n")
labels.foreach { l =>
  print_writer.write(s"Recall($l) = " + metrics_.recall(l) + "\n")
}
print_writer.write("\n")

val accuracy = metrics_.accuracy
val weightedFalsePositiveRate = metrics_.weightedFalsePositiveRate
val weightedFMeasure = metrics_.weightedFMeasure
val weightedPrecision = metrics_.weightedPrecision
val weightedRecall = metrics_.weightedRecall
val weightedTruePositiveRate = metrics_.weightedTruePositiveRate

print_writer.write("Sumário de métricas" + "\n")
print_writer.write(s"Acurácia = $accuracy" + "\n")
print_writer.write(s"weightedFalsePositiveRate = $weightedFalsePositiveRate" + "\n")
print_writer.write(s"weightedFMeasure = $weightedFMeasure" + "\n")
print_writer.write(s"weightedPrecision = $weightedPrecision" + "\n")
print_writer.write(s"weightedRecall = $weightedRecall" + "\n")
print_writer.write(s"weightedTruePositiveRate = $weightedTruePositiveRate" + "\n")

// Fechando o printwriter    
print_writer.close() 

//Salvar o modelo no HDFS
model.save("hdfs://master:8020/bigdata/model_random_forest")