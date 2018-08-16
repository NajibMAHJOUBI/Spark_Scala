package fr.spark.evaluation.clustering

import fr.spark.evaluation.utils.LoadDataTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.junit.{After, Before, Test}
import org.apache.spark.ml.linalg.DenseVector
import breeze.linalg.{DenseVector => BDV}

class ExplainedVarianceTaskTest {

  private var spark: SparkSession = _
  private var data: DataFrame = _


  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("explained variance test suite")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    data = new LoadDataTask("src/test/scala/resources/").run(spark, "features")


  }

  @Test def testBExplainedVarianceTask(): Unit = {
    data.show(false)

    val trans = data.rdd.map(row => row.getAs[DenseVector](row.fieldIndex("features")).toArray).map(p => new BDV(p)).reduce(_ + _)

    println(trans)
    println(trans.getClass)

    val number = data.count()
    println((1.0/number.toDouble)*trans)

  }

  @After def afterAll() {
    spark.stop()
  }

}



