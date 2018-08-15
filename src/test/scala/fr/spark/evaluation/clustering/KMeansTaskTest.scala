package fr.spark.evaluation.clustering

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row

class KMeansTaskTest {

  private var spark: SparkSession = _
  private val kClusters: Int = 2
  private val featuresColumn: String = "features"
  private val predictionColumn: String = "prediction"


  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)



    val data = Seq(Vectors.dense(Array(0.0, 0.0)),
                   Vectors.dense(Array(1.0, 1.0)),
                   Vectors.dense(Array(9.0, 8.0)),
                   Vectors.dense(Array(8., 9.0))).map(value => Row(value))

//      Vectors.dense([1.0, 1.0]),
//    Vectors.dense([9.0, 8.0]),
//    Vectors.dense([8.0, 9.0]))

 val df = spark.createDataFrame(data)


  }

  @Test def testKmeans(): Unit = {
    val kmeans = new KMeansTask(featuresColumn, predictionColumn)
    kmeans.defineModel(kClusters)
    
    assert(kmeans.kmeans.getK == kClusters)
    assert(kmeans.kmeans.getFeaturesColumn == featuresColumn)
    assert(kmeans.kmeans.getPredictionColumn == predictionColumn)
  }

  @After def afterAll() {
    spark.stop()
  }

}



