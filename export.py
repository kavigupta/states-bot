import os
from qgis.core import (QgsProject, QgsLayoutExporter, QgsApplication)

QgsApplication.setPrefixPath("/usr", True)

gui_flag = True
app = QgsApplication([], gui_flag)

app.initQgis()

project_path = os.path.abspath("statesmap.qgs")

project_instance = QgsProject.instance()
project_instance.setFileName(project_path)
project_instance.read()

manager = QgsProject.instance().layoutManager()
layout = manager.layoutByName("Main")

exporter = QgsLayoutExporter(layout)
exporter.exportToImage(project_instance.absolutePath() + "/layout.png",
                     QgsLayoutExporter.ImageExportSettings())

# app.exitQgis()
