package fr.spark.evaluation.clustering

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.junit.{After, Before, Test}

class BisectingKMeansTaskTest {

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

    val data = Seq(new DenseVector(Array(0.0, 0.0)),
      new DenseVector(Array(1.0, 1.0)),
      new DenseVector(Array(9.0, 8.0)),
      new DenseVector(Array(8.0, 9.0))).map(value => Row(value))
    val rdd = spark.sparkContext.parallelize(data)

    val schema = StructType(Seq(StructField("features", VectorType, false)))

    dataFrame = spark.createDataFrame(rdd, schema)
  }

  @Test def testBisectingKMeans(): Unit = {
    val bisectingKMeans = new BisectingKMeansTask(featuresColumn, predictionColumn)
    bisectingKMeans.defineModel(kClusters)

    assert(bisectingKMeans.bisectingKMeans.getK == kClusters)
    assert(bisectingKMeans.bisectingKMeans.getFeaturesCol == featuresColumn)
    assert(bisectingKMeans.bisectingKMeans.getPredictionCol == predictionColumn)

    bisectingKMeans.fit(dataFrame)
    assert(bisectingKMeans.model.getFeaturesCol == featuresColumn)
    assert(bisectingKMeans.model.getPredictionCol == predictionColumn)
    assert(bisectingKMeans.model.getK  == kClusters)

    val transform = bisectingKMeans.transform(dataFrame)
    assert(transform.isInstanceOf[DataFrame])
    assert(transform.count() == dataFrame.count())
    assert(transform.columns.contains(featuresColumn))
    assert(transform.columns.contains(predictionColumn))

    val cost = bisectingKMeans.computeCost(dataFrame)
    assert(cost.isInstanceOf[Double])
  }

  @After def afterAll() {
    spark.stop()
  }

}



