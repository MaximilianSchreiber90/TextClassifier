# TextClassifier

Data: Der Code geht davon aus, dass ein Data Verzeichnis existiert in dem ein paar Daten zu finden sind. Daten sind in GoogleDrive

Binary.ipynb: Letzte Version mit der ich experimentiert habe um Ergebnisse für Binary Classification zu gewinnen. Alle relevanten Infos sind jedoch auch in Final Pipeline.ipynb enthalten

FinalPipeline.ipynb: Ruft alle relevanten Funktionen auf um die Binary Evaluation zu rekonstruieren und sich eine Liste der besten Classifier zu erstellen

OptimizingLogReg.ipynb: Kann genutzt werden um den Optimalen Threshold zu bestimmen. Da nicht jeder LinearSVM keinen Threshold hat und die Auswirkungen kaum ins Gewicht fallen, wurde dieser Teil nicht in die FinalEvaluation.ipynb übernommen

Reuters.ipynb: Enthält explorative Versuche auf die Reuters Daten zuzugreifen. Aus nltk.corpus kann auf sie zugegriffen werden. Für sklearn.datasets gestaltet es sich schwerer. Sklearn 1.20+ könnte hier helfen, da es ab dieser Version zu jedem Datensatz eine DESCRiption gibt

TestingClassifiers.ipynb: Enthält alle Erkenntnisse der Multilabeling Classifications mittels OneVsRest. Es enthält die Pipeline mit der der reste Vergleich der Classifiers erstellt wurde

access_data.py: enthält alle Funktionen die benutzt werden um auf die verschiedenen Datensätze (Reuters, Huffingpost, 20 Newsgroup) zuzugreifen und sie in ein einheitlichen Dataframe zu vermischen.

evaluating_data.py: enthällt alle Funktionen die zur Evaluation geschrieben wurden

feature_engineering.py: Wird in der FinalPipeline.ipynb nicht genutzt, da sich ein Feature Engineering als wenig Effektiv gezeigt hat. Die Verwendung von TruncatedSVD ist daher einfacher und einheitlicher. Aber selbst diese verbessert die Ergebnisse nicht immer und wenn nur kaum, benötigt aber teils wesentlich mehr Zeit

Preparing_data.py: enthält Funktionen die zum Splitten und Vorbereiten der Daten verwendet werden
