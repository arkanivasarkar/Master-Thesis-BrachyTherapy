import numpy as np
import vtk
import slicer

# --- Get nodes ---
volumeNode = slicer.util.getNode('vtkMRMLScalarVolumeNode*')  # Replace with exact name if needed
lineNode = slicer.util.getNode('L')  # Your line markup

# --- Sample intensity values along the line ---
def sampleLineIntensity(volumeNode, lineNode, numSamples=100):
    imageData = volumeNode.GetImageData()

    # RAS to IJK
    rasToIjkMatrix = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(rasToIjkMatrix)

    p1 = np.array(lineNode.GetNthControlPointPositionVector(0))
    p2 = np.array(lineNode.GetNthControlPointPositionVector(1))
    pointsRAS = [p1 + (p2 - p1) * i / (numSamples - 1) for i in range(numSamples)]

    intensities = []
    for ras in pointsRAS:
        ras_hom = list(ras) + [1]
        ijk = [int(round(sum(rasToIjkMatrix.GetElement(i, j) * ras_hom[j] for j in range(4)))) for i in range(3)]
        dims = volumeNode.GetImageData().GetDimensions()
        if all(0 <= ijk[i] < dims[i] for i in range(3)):
            val = imageData.GetScalarComponentAsDouble(ijk[0], ijk[1], ijk[2], 0)
            intensities.append(val)
        else:
            intensities.append(0)
    return intensities

intensities = sampleLineIntensity(volumeNode, lineNode, numSamples=200)

# --- Create table ---
tableNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTableNode', 'LineIntensityTable')
table = tableNode.GetTable()

xArray = vtk.vtkFloatArray()
xArray.SetName("Position Index")

yArray = vtk.vtkFloatArray()
yArray.SetName("Intensity")

for i, val in enumerate(intensities):
    xArray.InsertNextValue(i)
    yArray.InsertNextValue(val)

table.AddColumn(xArray)
table.AddColumn(yArray)
table.SetNumberOfRows(len(intensities))

# --- Create plot series node ---
plotSeriesNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLPlotSeriesNode', 'LineIntensitySeries')
plotSeriesNode.SetAndObserveTableNodeID(tableNode.GetID())
plotSeriesNode.SetXColumnName("Position Index")
plotSeriesNode.SetYColumnName("Intensity")
plotSeriesNode.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeLine)
plotSeriesNode.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone)
plotSeriesNode.SetColor(0.2, 0.6, 1.0)

# --- Create plot chart node ---
chartNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLPlotChartNode', 'LineIntensityChart')
chartNode.AddAndObservePlotSeriesNodeID(plotSeriesNode.GetID())

# --- Show the plot in layout ---
slicer.modules.plots.logic().ShowChartInLayout(chartNode)
