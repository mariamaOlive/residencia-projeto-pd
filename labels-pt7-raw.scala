import java.util.Base64
import java.nio.charset.StandardCharsets
import org.apache.spark.sql.types.{StructField, StructType, StringType, ShortType} 

val myManualSchema = StructType(Array(
    StructField("label", StringType, false),
    StructField("url", StringType, false),
    StructField("hash", StringType, false),
    StructField("html_base64", StringType, false)
  ))

val pt7 = { 
	spark.read.format("csv")
	.option("header", "false")
	.option("delimiter", "\t")
	.schema(myManualSchema)
	.load("hdfs://master:8020/bigdata/pt7-raw")
}

val tldDF = { 
	pt7.select("url", "html_base64").map(row => {
			// row.getAs[String](0) => url
			// row.getAs[String](1) => base64 html

			// http://www.xxxx.com.br/ => parts(2) => .br
			val parts = row.getAs[String](0).split("/")
            val country = parts(2)
			val reg = """\.([a-z]{2,3})\/?$""".r
			(reg.findAllIn(country).mkString, row.getAs[String](0), row.getAs[String](1))})
	.withColumnRenamed("_1", "label")
	.withColumnRenamed("_2", "url")
	.withColumnRenamed("_3", "text64byte")
	.filter("label != ''")
}

tldDF.write.mode("overwrite").format("parquet").save("/bigdata/pt7-multilabel")
