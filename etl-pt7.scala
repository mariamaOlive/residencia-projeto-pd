import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

import spark.implicits._

// schema (pt7-raw)
// _c0 = label
// _c1 = url
// _c3 = raw text em codificação base64

// lê os dados do HDFS e converte para um dataframe
val df = { 
	spark.read
	.format("parquet")
	.load("hdfs://master:8020/bigdata/pt7-multilabel")
	.withColumnRenamed("_c0","label")
	.withColumnRenamed("_c1","url")
	.withColumnRenamed("_c3","text64byte")
}

// elimina textos nulos
// converte base64 em txt
// elimina as colunas em base64, o digest e a url
// esquema final => label / raw (texto de cada página web)
val raw_df = {
	df.filter("text64byte is not null")
	.withColumn("txt" , unbase64(col("text64byte"))
	.cast("string"))
	.drop("text64byte")
}

// processa tokenizer sobre o texto da página web
val tokenizer = { new RegexTokenizer()
	.setGaps(false)
	.setPattern("[\\p{L}\\w&&[^\\d]]+")
	.setMinTokenLength(4)
	.setInputCol("txt")
	.setOutputCol("tokens")
}

// remove stopwords 
val stopWordsRemover = {
	new StopWordsRemover()
	.setStopWords(StopWordsRemover.loadDefaultStopWords("portuguese"))
	.setInputCol("tokens")
	.setOutputCol("filtered")
}

// processa TD-IDF sobre cada palavra de cada página web
// atribuindo um valor a cada palavra na coleção 
// ao final, cria um vetor esparso de 2^18 características para cada página
val hashingTF = { 
	new HashingTF()
	.setInputCol("filtered")
	.setOutputCol("features-hash")
}

val idf = new IDF().setInputCol("features-hash").setOutputCol("features")

// cria o pipeline com todos os estágios
val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf))

// ajusta os dados com base nos estágios do pipeline
val model = pipeline.fit(raw_df)
val result = model.transform(raw_df)

// salva os dados processados no HDFS
val columnNames = Seq("label","features")
val the_df = result.select(columnNames.head, columnNames.tail: _*)
the_df.write.mode("overwrite").save("hdfs://master:8020/bigdata/pt7-hash.parquet")