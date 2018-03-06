
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{udf, col}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.clustering.KMeans


object ChIndexExample {

   def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("ChIndexExample")
      .getOrCreate()


    val createFeatures = udf((x: Double, y: Double) => {Vectors.dense(x, y)})

    val path = "data/clustering"
    var data = spark.read.parquet(path)
                .withColumn("features", createFeatures(col("x"), col("y")))
                .persist()
    
   
   val kmeans = new KMeans()
                 .setFeaturesCol("features")
                 .setK(16)
                 .setMaxIter(20)
                 .setPredictionCol("prediction")
                 .setSeed(123L)
                 .setTol(0.0001)
   val model = kmeans.fit(data)
   val prediction = model.transform(data)
   prediction.show()
   model.clusterCenters.foreach(println) 
    spark.stop()
  }





}
