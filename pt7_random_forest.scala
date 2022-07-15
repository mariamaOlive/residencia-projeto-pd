import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import spark.implicits._


// lê os dados do HDFS e converte para um dataframe
val df = { 
	spark.read
	.format("parquet")
	.load("hdfs://master:8020/bigdata/pt7-hash.parquet")
}


// //Indexando coluna de labels
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(df)


// Split the data into training and test sets (30% held out for testing).
// val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), 42)

val Array(trainingData, testData) = df.randomSplit(Array(0.01, 0.3), 42)
//treino, teste = a.randomSplit(weights = [0.7, 0.3], seed = SEED)

// Instanciando o classificador - Random Forest
val rf = new RandomForestClassifier()
		.setLabelCol("indexedLabel")
		.setFeaturesCol("features")
	    .setNumTrees(10)


// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labelsArray(0))

// Chain indexers and forest in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, rf, labelConverter))

//TODO: Remover caso nao utilizar CV
// val paramGrid = new ParamGridBuilder()
// 				.addGrid(rf.numTrees, Array(10, 20))
// 				.build()

// val cv = new CrossValidator()
// 		.setEstimator(pipeline)
// 		.setEvaluator(new MulticlassClassificationEvaluator())
// 		.setEstimatorParamMaps(paramGrid)
// 		.setNumFolds(3)

// //
// val cvModel = cv.fit(trainingData)


// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)


//Funcao responsavel por calcular as metricas
def cal_metrics(metric: String) : Double = {
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName(metric)

  val metric_value = evaluator.evaluate(predictions)
  return metric_value
}



//Metrics 
//accuracy
//f1
//precisionByLabel
//weightedPrecision
//recallByLabel
//weightedRecall
val metrics = Array("accuracy", "f1", "precisionByLabel", "weightedPrecision", "recallByLabel", "weightedRecall")

//Calcular metricas 
// foreach metrics


// foreach(f : scala.Function1[T, scala.Unit]) : scala.Unit
//Salvar em arquivo as métricas
import scala.collection.mutable.ArrayBuffer
val final_metrics  = ArrayBuffer[(String, Double)]()

for(metric <- metrics){
  print(metric)
  // final_metrics = final_metrics :+ cal_metrics(metric)
  final_metrics += ((metric, cal_metrics(metric)))
}

//Salvar métrica no arquivo

//Salvar o modelo no HDFS








//Deixa em uma variavel somente label e prediction
// val predictionAndLabels = predictions.select("indexedLabel", "prediction")

// val metrics = new MulticlassMetrics(predictionAndLabels.rdd)

// // Confusion matrix
// println("Confusion matrix:")
// println(metrics.confusionMatrix)

// // Overall Statistics
// val accuracy = metrics.accuracy
// println("Summary Statistics")
// println(s"Accuracy = $accuracy")

// // Precision by label
// val labels = metrics.labels
// labels.foreach { l =>
//   println(s"Precision($l) = " + metrics.precision(l))
// }

// // Recall by label
// labels.foreach { l =>
//   println(s"Recall($l) = " + metrics.recall(l))
// }

// // False positive rate by label
// labels.foreach { l =>
//   println(s"FPR($l) = " + metrics.falsePositiveRate(l))
// }

// // F-measure by label
// labels.foreach { l =>
//   println(s"F1-Score($l) = " + metrics.fMeasure(l))
// }

// // Weighted stats
// println(s"Weighted precision: ${metrics.weightedPrecision}")
// println(s"Weighted recall: ${metrics.weightedRecall}")
// println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
// println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")

//Não sei o que é isso --> Descobrir!!
// val rfModel = cvModel.stages(2).asInstanceOf[RandomForestClassificationModel]
// println(s"Learned classification forest model:\n ${rfModel.toDebugString}")


// // Train model. This also runs the indexers.
// val model = pipeline.fit(trainingData)

// // Make predictions.
// val predictions = model.transform(testData)

// // Select example rows to display.
// predictions.select("predictedLabel", "label", "features").show(5)

// // Select (prediction, true label) and compute test error.
// val evaluator = new MulticlassClassificationEvaluator()
//   .setLabelCol("indexedLabel")
//   .setPredictionCol("prediction")
//   .setMetricName("accuracy")
// val accuracy = evaluator.evaluate(predictions)
// println(s"Test Error = ${(1.0 - accuracy)}")

// val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
// println(s"Learned classification forest model:\n ${rfModel.toDebugString}")