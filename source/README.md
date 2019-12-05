# Running VT-LinKEr in your local machine

To run **VT-LinKEr** in your local machine, please follow these steps.

**Step 1:**

Download the required [files](https://figshare.com/articles/VTKEL_resource_files/8247770/3).

**Step 2:**

Download python source [code](https://github.com/shahidost/Baseline4VTKEL/tree/master/source/code).

**Step 3:**

Save the source code and required files in a directory ($root).

**Step 4:**

Update python file in your local machine.

1. [main_VT-LinKEr_System.py](https://github.com/shahidost/Baseline4VTKEL/blob/master/source/code/main_VT-LinKEr_System.py)

-	Line 461: insert path for YAGO taxonomy .ttl file.
-	Line 470: insert image (images should be downloaded from Flickr30k-entities dataset) directory path.
-	Line 486: insert image captions directory path.
-	Line 513 & 516: insert YOLO pre-train files path.
-	Line 633: insert directory path to save annotated image-files from VT-LinKEr.
-	Line 637: insert directory path to save Turtle annotated file from VT-LinKEr.

2. Run *main_VT-LinKEr_System.py* by using the command for VT-LinKEr:
```
> python $root/code/main_VT-LinKEr_System.py
```
3. You can find the annotated output *.ttl* and *.jpg* files in your local machine.

4. The debugging/execution of VT-LinKEr will take some time due to the processing of YAGO-ontology w.r.t to the speed of your machine.
