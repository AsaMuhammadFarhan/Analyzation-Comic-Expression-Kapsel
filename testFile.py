expressionFinalData = ["Angry", "Happy", "Surprised"]
pageFinalData = [0,1,2]

## Ganti nama orang ketika ganti orang
personName = "KrisnaSanjayyay"
fileName = "ExpressionLog.txt"
def finalResult():
  f = open(fileName, "a")
  f.writelines(personName + "'s expression:\n")
  for x in range(len(pageFinalData)):
    f.writelines("Hal " + str(pageFinalData[x]) + ": " + expressionFinalData[x] + "\n")
  f.writelines("\n")
  f.writelines("\n")
  f.close()

finalResult()
