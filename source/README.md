# Running VTKEL System in your local machine

To run **VTKEL Baseline system** in your local machine follow below steps carefully.

**Step 1:**

Download the required [files](https://figshare.com/articles/VTKEL_resource_files/8247770/3) to run the source code of VTKEL system.

**Step 2:**

Download python source code.

**Step 3:**

Save the source code and required files in a directory.

**Step 4:**

Update below python files in your local machine.

1. main_VTKEL_Baseline_System.py

-	Line 133: insert the path of image caption file downloaded from Flickr30k entities dataset.
-	Line 364: insert the path of YOLO pre-trained classes file downloaded from YOLO files.
-	Line 367: insert the path of YOLO weights and .cfg files downloaded from YOLO files.
-	Line 808: insert file path to store the annotated .ttl file in your local machine.
-	Line 811: insert file path to store the annotated .jpg file in your local machine.
-	Line 77: change the number *85* according to the directory path where you want to store the image(s). Count the string of directory path characters from root directory till image file (.i.e. [C:/files/images/]1234.jpg [16.jpg]).

2. VTKEL_annotations.py
-	Line 27: insert the path of image caption [.txt] file downloaded from Flickr30k entities dataset (as described in Line:133 of 'main_VTKEL_Baseline_System.py'.

3. Bounding_boxes_annotations.py
-	Line 39: change as according to guideline of 'main_VTKEL_Baseline_System.py' line 77.

4. Coreference_rdfTripes_from_PIKES
-	Line 36: change as according to guideline of 'main_VTKEL_Baseline_System.py' line 77.

5. YAGO_Taxonomy.py
-	Line 25: insert the path of YAGO taxonomy file downloaded from YAGO files.
-	Line 30: insert the path of image to be uploaded for processing (downloaded from Flickr30k entities dataset).
