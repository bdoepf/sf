package com.github.bdoepf.sf

import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.{PipelineStage, Transformer}


class MLPModel(numOfinputFeatures: Int, numberOfLabels: Int) extends SfModel {
  private val mlp = new MultilayerPerceptronClassifier()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("features")
    .setBlockSize(128)
    .setSeed(1234L)
    .setSolver("gd")

  override def model(): PipelineStage = mlp


  override def addGridParams(builder: ParamGridBuilder): ParamGridBuilder = {
    builder
      .addGrid(mlp.layers, Array(Array[Int](numOfinputFeatures.toInt, 5, 10, 20, numberOfLabels)))
      .addGrid(mlp.maxIter, Array(500))
  }

  override def evaluate(transformer: Transformer): Unit = transformer match {
    case model: MultilayerPerceptronClassificationModel =>
      model.save(s"model/MultilayerPerceptronClassificationModel_${System.currentTimeMillis()}")
      println(
        s"""MultilayerPerceptronClassificationModel:
           | layers: ${model.layers.toList}
           | weights: ${model.weights}""".stripMargin
      )
    case _ => println("Best model isn't a MultilayerPerceptronClassificationModel.")
  }
}
