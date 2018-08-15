package fr.spark.evaluation.clustering

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.{ArrayType, DoubleType, _}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.junit.{After, Before, Test}

class KMeansTaskTest {

  private var spark: SparkSession = _
  private val kClusters: Int = 2
  private val featuresColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private var dataFrame: DataFrame = _


  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("KMeans test suite")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    val data = Seq(Vectors.dense(Array(0.0, 0.0)),
                   Vectors.dense(Array(1.0, 1.0)),
                   Vectors.dense(Array(9.0, 8.0)),
                   Vectors.dense(Array(8.0, 9.0))).map(value => Row(value))
    val rdd = spark.sparkContext.parallelize(data)

    val schema = StructType(Seq(StructField("features", new VectorUDT(), false)))

    dataFrame = spark.createDataFrame(rdd, schema)
  }

  @Test def testKMeans(): Unit = {
    val kmeans = new KMeansTask(featuresColumn, predictionColumn)
    kmeans.defineModel(kClusters)
    
    assert(kmeans.kMeans.getK == kClusters)
    assert(kmeans.kMeans.getFeaturesCol == featuresColumn)
    assert(kmeans.kMeans.getPredictionCol == predictionColumn)

    dataFrame.show()
  }

  @After def afterAll() {
    spark.stop()
  }

}



