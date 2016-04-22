import xml.etree.cElementTree as ET





def creator():
	classes = "acacia", "trompa", "balela"
	root = ET.Element("roi_coordinates")


	cls_list = []
	#create a field for each class's roi_coordinates
	for cls in classes:
		curr_class = ET.SubElement(root, cls)

		# xmlET.SubElement(root, cls)
		cls_list.append(curr_class)
	return (root, cls_list)

def manolo(root, classes):
	inteiro = 69
	coordXY = ET.SubElement(classes[0], ("_coordinates_"+ str(inteiro)))
	
	ET.SubElement(coordXY, "X").text = "800"
	ET.SubElement(coordXY, "Y").text = "600"
				
	#ET.SubElement(classes[0], point1) = coordXY	

	#ET.SubElement(classes[0], "field1", name="blah").text = "some value1"
	ET.SubElement(classes[1], "field2", name="asdfasd").text = "some vlaue2"



	tree = ET.ElementTree(root)
	tree.write("filename.xml")


rt, classes = creator()

manolo(rt, classes)
