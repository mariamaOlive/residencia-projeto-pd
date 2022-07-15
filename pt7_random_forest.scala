import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import spark.implicits._

import java.io.File
import java.io.PrintWriter


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
// val labelConverter = new IndexToString()
//   .setInputCol("prediction")
//   .setOutputCol("predictedLabel")
//   .setLabels(labelIndexer.labelsArray(0))  ///Taa dando erro aqui

// Chain indexers and forest in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, rf))

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

//////////Funcionando ate aqui/////////////////


// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("label", "features").show(5)

//Funcao responsavel por calcular as metricas
// def cal_metrics(metric: String, predictions: org.apache.spark.sql.Dataset[_]) : Double = {
//   val evaluator = new MulticlassClassificationEvaluator()
//     .setLabelCol("indexedLabel")
//     .setPredictionCol("prediction")
//     .setMetricName(metric)

//   val metric_value = evaluator.evaluate(predictions)
//   return metric_value
// }



//Metrics 
//accuracy
//f1
//precisionByLabel
//weightedPrecision
//recallByLabel
//weightedRecall
// val metrics = Array("accuracy", "f1", "precisionByLabel", "weightedPrecision", "recallByLabel", "weightedRecall")

//Calcular metricas 

//Salvar em arquivo as métricas
import scala.collection.mutable.ArrayBuffer
val final_metrics  = ArrayBuffer[(String, Double)]()

// for(metric <- metrics){
//   print(metric)
//   // final_metrics = final_metrics :+ cal_metrics(metric)
//   final_metrics += ((metric, cal_metrics(metric, predictions )))
// }


//select only prediction and label columns
val preds_and_labels = predictions.select("prediction","indexedLabel")

val metrics_ = new MulticlassMetrics(preds_and_labels.rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

//Escrever metricas no arquivo
val file_Object = new File("/user_data/metrics.txt" ) 
val print_writer = new PrintWriter(file_Object) 

// Confusion Matrix
val confusionMatrixOutput = metrics_.confusionMatrix.toString
// val confusionMatrixOutputFinal = spark.parallelize(confusionMatrixOutput)
print_writer.write(s"ConfusionMatrix:\n$confusionMatrixOutput")
print_writer.write("\n\n")

// Labels
val labels = metrics_.labels

// False positive rate by label
labels.foreach { l =>
  print_writer.write(s"FPR($l) = " + metrics_.falsePositiveRate(l) + "\n")
}
print_writer.write("\n")

// True positive rate by label
labels.foreach { l =>
  print_writer.write(s"TPR($l) = " + metrics_.truePositiveRate(l) + "\n")
}
print_writer.write("\n")

// F-measure by label
labels.foreach { l =>
  print_writer.write(s"F1-Score($l) = " + metrics_.fMeasure(l) + "\n")
}
print_writer.write("\n")

// Precision by label
labels.foreach { l =>
  print_writer.write(s"Precision($l) = " + metrics_.precision(l) + "\n")
}
print_writer.write("\n")

// Recall by label
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

print_writer.write("Summary Statistics" + "\n")
print_writer.write(s"Accuracy = $accuracy" + "\n")
print_writer.write(s"weightedFalsePositiveRate = $weightedFalsePositiveRate" + "\n")
print_writer.write(s"weightedFMeasure = $weightedFMeasure" + "\n")
print_writer.write(s"weightedPrecision = $weightedPrecision" + "\n")
print_writer.write(s"weightedRecall = $weightedRecall" + "\n")
print_writer.write(s"weightedTruePositiveRate = $weightedTruePositiveRate" + "\n")




// accuracy          falsePositiveRate   logLoss     truePositiveRate            weightedPrecision
// confusionMatrix   hammingLoss         precision   weightedFMeasure            weightedRecall
// fMeasure          labels              recall      weightedFalsePositiveRate   weightedTruePositiveRate


// Closing printwriter    
print_writer.close() 


//Salvar o modelo no HDFS
// model.save("hdfs://master:8020/bigdata/modelo_rf")







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