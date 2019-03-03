package com.github.bdoepf.sf

import org.apache.spark.ml.{PipelineStage, Transformer}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.ParamGridBuilder

class RfModel(labelColName: String = "indexedLabel", featureColName: String = "features") extends SfModel {

  private val rf = new RandomForestClassifier()
    .setLabelCol(labelColName)
    .setFeaturesCol(featureColName)

  override def model(): PipelineStage = rf

  override def addGridParams(builder: ParamGridBuilder): ParamGridBuilder = {
    builder.addGrid(rf.numTrees, Array(30))
      .addGrid(rf.maxDepth, Array(5, 15))
      .addGrid(rf.maxBins, Array(40))
  }

  override def evaluate(bestModel: Transformer): Unit = {
    bestModel match {
      case model: RandomForestClassificationModel =>
        model.save(s"model/RandomForestClassificationModel_${System.currentTimeMillis()}")
        println(
          s"""RandomForest classification tree model:
             | numTrees: ${model.getNumTrees}
             | totalNumNodes: ${model.totalNumNodes}
             | maxDepth: ${model.getMaxDepth}
             | maxBins: ${model.getMaxBins}""".stripMargin
        )
      case _ => println("Best model isn't a RandomForestClassificationModel.")
    }
  }
}
