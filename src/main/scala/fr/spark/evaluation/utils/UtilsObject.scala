package fr.spark.evaluation.utils

import org.apache.spark.ml.linalg.{Vector, Vectors}

import scala.collection.mutable.WrappedArray


object UtilsObject {

  def defineDenseVector(values: WrappedArray[Double]): Vector = {
    Vectors.dense(values.toArray)
  }

}
