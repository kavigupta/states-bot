import os
import sys

from qgis.core import QgsProject, QgsLayoutExporter, QgsApplication


_, title, path, dpi = sys.argv
dpi = int(dpi)

QgsApplication.setPrefixPath("/usr", True)

gui_flag = False
app = QgsApplication([], gui_flag)

app.initQgis()

project_path = os.path.abspath("statesmap.qgs")

project_instance = QgsProject.instance()
project_instance.setFileName(project_path)
project_instance.read()
project_instance.setTitle(title)

manager = QgsProject.instance().layoutManager()
layout = manager.layoutByName("Main")

exporter = QgsLayoutExporter(layout)
settings = QgsLayoutExporter.ImageExportSettings()
settings.dpi = dpi
exporter.exportToImage(
    os.path.abspath(path),
    settings
)
app.exitQgis()
