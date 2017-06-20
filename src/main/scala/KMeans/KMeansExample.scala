
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.clustering.{KMeans => MLlibKMeans}
import org.apache.spark.sql.functions.{udf, col}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.sql.SparkSession


object KMeansExample {


  def defineKMeans(nbCluster: Int): KMeans = {
    new KMeans()
        .setK(nbCluster)
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setMaxIter(20)
        .setTol(0.0001)
        .setInitMode(MLlibKMeans.K_MEANS_PARALLEL)
        .setInitSteps(20)
        .setSeed(1L)
  }


  val append = udf((x: Double, y: Double) => Vectors.dense(x, y))


  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("KMeans")
      .getOrCreate()


    val path = "data/clustering"
    var data = spark.read.parquet(path)
    data = data.select(append(col("x"), col("y")).alias("features"))
    data.show(5)

    // Trains a k-means model.
    
    val prediction = defineKMeans(16).fit(data).transform(data)
    prediction.write.parquet("target/model/kmeans")

    spark.stop()
  }
}
// scalastyle:on println
