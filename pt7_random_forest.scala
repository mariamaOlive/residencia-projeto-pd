import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

import spark.implicits._


// lê os dados do HDFS e converte para um dataframe
val df = { 
	spark.read
	.format("parquet")
	.load("hdfs://master:8020/bigdata/pt7-hash.parquet")
	.withColumnRenamed("_1","label")
	.withColumnRenamed("_2","url")
	.withColumnRenamed("_3","words")
}

val random = Math.abs(scala.util.Random.nextInt)

// //Indexando coluna de labels
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(df)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), random)

// Instanciando o classificador - Random Forest
val rf = new RandomForestClassifier()
		.setLabelCol("indexedLabel")
		.setFeaturesCol("features")
//   .setNumTrees(10)


// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labelsArray(0))

// Chain indexers and forest in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, rf, labelConverter))

val paramGrid = new ParamGridBuilder()
				.addGrid(rf.numTrees, Array(10, 20, 30, 40, 50))
				.build()

val cv = new CrossValidator()
		.setEstimator(pipeline)
		.setEvaluator(new MulticlassClassificationEvaluator())
		.setEstimatorParamMaps(paramGrid)
		.setNumFolds(10)


val cvModel = cv.fit(trainingData)

val results = cvModel
            .transform(testData)
            .select("prediction", "label")



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