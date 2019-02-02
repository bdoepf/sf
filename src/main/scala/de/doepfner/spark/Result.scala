package de.doepfner.spark

final case class ResultItem(Id: Int, predictedLabel: String)


final case class SubmissionItem(Id: Int,
                                ARSON: Int = 0,
                                ASSAULT: Int = 0,
                                `BAD CHECKS`: Int = 0,
                                BRIBERY: Int = 0,
                                BURGLARY: Int = 0,
                                `DISORDERLY CONDUCT`: Int = 0,
                                `DRIVING UNDER THE INFLUENCE`: Int = 0,
                                `DRUG/NARCOTIC`: Int = 0,
                                DRUNKENNESS: Int = 0,
                                EMBEZZLEMENT: Int = 0,
                                EXTORTION: Int = 0,
                                `FAMILY OFFENSES`: Int = 0,
                                `FORGERY/COUNTERFEITING`: Int = 0,
                                FRAUD: Int = 0,
                                GAMBLING: Int = 0,
                                KIDNAPPING: Int = 0,
                                `LARCENY/THEFT`: Int = 0,
                                `LIQUOR LAWS`: Int = 0,
                                LOITERING: Int = 0,
                                `MISSING PERSON`: Int = 0,
                                `NON-CRIMINAL`: Int = 0,
                                `OTHER OFFENSES`: Int = 0,
                                `PORNOGRAPHY/OBSCENE MAT`: Int = 0,
                                PROSTITUTION: Int = 0,
                                `RECOVERED VEHICLE`: Int = 0,
                                ROBBERY: Int = 0,
                                RUNAWAY: Int = 0,
                                `SECONDARY CODES`: Int = 0,
                                `SEX OFFENSES FORCIBLE`: Int = 0,
                                `SEX OFFENSES NON FORCIBLE`: Int = 0,
                                `STOLEN PROPERTY`: Int = 0,
                                SUICIDE: Int = 0,
                                `SUSPICIOUS OCC`: Int = 0,
                                TREA: Int = 0,
                                TRESPASS: Int = 0,
                                VANDALISM: Int = 0,
                                `VEHICLE THEFT`: Int = 0,
                                WARRANTS: Int = 0,
                                `WEAPON LAWS`: Int = 0
                               )

object Result {

  def convertToSubmission(resultItem: ResultItem): SubmissionItem = {
    resultItem.predictedLabel match {
      case "ARSON" => SubmissionItem(resultItem.Id, ARSON = 1)
      case "ASSAULT" => SubmissionItem(resultItem.Id, ASSAULT = 1)
      case "BAD CHECKS" => SubmissionItem(resultItem.Id, `BAD CHECKS` = 1)
      case "BRIBERY" => SubmissionItem(resultItem.Id, BRIBERY = 1)
      case "BURGLARY" => SubmissionItem(resultItem.Id, BURGLARY = 1)
      case "DISORDERLY CONDUCT" => SubmissionItem(resultItem.Id, `DISORDERLY CONDUCT` = 1)
      case "DRIVING UNDER THE INFLUENCE" => SubmissionItem(resultItem.Id, `DRIVING UNDER THE INFLUENCE` = 1)
      case "DRUG/NARCOTIC" => SubmissionItem(resultItem.Id, `DRUG/NARCOTIC` = 1)
      case "DRUNKENNESS" => SubmissionItem(resultItem.Id, DRUNKENNESS = 1)
      case "EMBEZZLEMENT" => SubmissionItem(resultItem.Id, EMBEZZLEMENT = 1)
      case "EXTORTION" => SubmissionItem(resultItem.Id, EXTORTION = 1)
      case "FAMILY OFFENSES" => SubmissionItem(resultItem.Id, `FAMILY OFFENSES` = 1)
      case "FORGERY/COUNTERFEITING" => SubmissionItem(resultItem.Id, `FORGERY/COUNTERFEITING` = 1)
      case "FRAUD" => SubmissionItem(resultItem.Id, FRAUD = 1)
      case "GAMBLING" => SubmissionItem(resultItem.Id, GAMBLING = 1)
      case "KIDNAPPING" => SubmissionItem(resultItem.Id, KIDNAPPING = 1)
      case "LARCENY/THEFT" => SubmissionItem(resultItem.Id, `LARCENY/THEFT` = 1)
      case "LIQUOR LAWS" => SubmissionItem(resultItem.Id, `LIQUOR LAWS` = 1)
      case "LOITERING" => SubmissionItem(resultItem.Id, LOITERING = 1)
      case "MISSING PERSON" => SubmissionItem(resultItem.Id, `MISSING PERSON` = 1)
      case "NON-CRIMINAL" => SubmissionItem(resultItem.Id, `NON-CRIMINAL` = 1)
      case "OTHER OFFENSES" => SubmissionItem(resultItem.Id, `OTHER OFFENSES` = 1)
      case "PORNOGRAPHY/OBSCENE MAT" => SubmissionItem(resultItem.Id, `PORNOGRAPHY/OBSCENE MAT` = 1)
      case "PROSTITUTION" => SubmissionItem(resultItem.Id, PROSTITUTION = 1)
      case "RECOVERED VEHICLE" => SubmissionItem(resultItem.Id, `RECOVERED VEHICLE` = 1)
      case "ROBBERY" => SubmissionItem(resultItem.Id, ROBBERY = 1)
      case "RUNAWAY" => SubmissionItem(resultItem.Id, RUNAWAY = 1)
      case "SECONDARY CODES" => SubmissionItem(resultItem.Id, `SECONDARY CODES` = 1)
      case "SEX OFFENSES FORCIBLE" => SubmissionItem(resultItem.Id, `SEX OFFENSES FORCIBLE` = 1)
      case "SEX OFFENSES NON FORCIBLE" => SubmissionItem(resultItem.Id, `SEX OFFENSES NON FORCIBLE` = 1)
      case "STOLEN PROPERTY" => SubmissionItem(resultItem.Id, `STOLEN PROPERTY` = 1)
      case "SUICIDE" => SubmissionItem(resultItem.Id, SUICIDE = 1)
      case "SUSPICIOUS OCC" => SubmissionItem(resultItem.Id, `SUSPICIOUS OCC` = 1)
      case "TREA" => SubmissionItem(resultItem.Id, TREA = 1)
      case "TRESPASS" => SubmissionItem(resultItem.Id, TRESPASS = 1)
      case "VANDALISM" => SubmissionItem(resultItem.Id, VANDALISM = 1)
      case "VEHICLE THEFT" => SubmissionItem(resultItem.Id, `VEHICLE THEFT` = 1)
      case "WARRANTS" => SubmissionItem(resultItem.Id, WARRANTS = 1)
      case "WEAPON LAWS" => SubmissionItem(resultItem.Id, `WEAPON LAWS` = 1)
    }

  }
}
