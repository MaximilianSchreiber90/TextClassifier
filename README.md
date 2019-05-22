# TextClassifier

<b>Data</b>: Der Code geht davon aus, dass ein Data Verzeichnis existiert in dem ein paar Daten zu finden sind. Daten sind in GoogleDrive

<b>Binary.ipynb</b>: Letzte Version mit der ich experimentiert habe um Ergebnisse für Binary Classification zu gewinnen. Alle relevanten Infos sind jedoch auch in Final Pipeline.ipynb enthalten

<b>FinalPipeline.ipynb</b>: Ruft alle relevanten Funktionen auf um die Binary Evaluation zu rekonstruieren und sich eine Liste der besten Classifier zu erstellen

<b>OptimizingLogReg.ipynb</b>: Kann genutzt werden um den Optimalen Threshold zu bestimmen. Da nicht jeder LinearSVM keinen Threshold hat und die Auswirkungen kaum ins Gewicht fallen, wurde dieser Teil nicht in die FinalEvaluation.ipynb übernommen

<b>Reuters.ipynb</b>: Enthält explorative Versuche auf die Reuters Daten zuzugreifen. Aus nltk.corpus kann auf sie zugegriffen werden. Für sklearn.datasets gestaltet es sich schwerer. Sklearn 1.20+ könnte hier helfen, da es ab dieser Version zu jedem Datensatz eine DESCRiption gibt

<b>TestingClassifiers.ipynb</b>: Enthält alle Erkenntnisse der Multilabeling Classifications mittels OneVsRest. Es enthält die Pipeline mit der der reste Vergleich der Classifiers erstellt wurde

<b>access_data.py</b>: enthält alle Funktionen die benutzt werden um auf die verschiedenen Datensätze (Reuters, Huffingpost, 20 Newsgroup) zuzugreifen und sie in ein einheitlichen Dataframe zu vermischen.

<b>evaluating_data.py</b>: enthällt alle Funktionen die zur Evaluation geschrieben wurden

<b>feature_engineering.py</b>: Wird in der FinalPipeline.ipynb nicht genutzt, da sich ein Feature Engineering als wenig Effektiv gezeigt hat. Die Verwendung von TruncatedSVD ist daher einfacher und einheitlicher. Aber selbst diese verbessert die Ergebnisse nicht immer und wenn nur kaum, benötigt aber teils wesentlich mehr Zeit

<b>Preparing_data.py</b>: enthält Funktionen die zum Splitten und Vorbereiten der Daten verwendet werden
