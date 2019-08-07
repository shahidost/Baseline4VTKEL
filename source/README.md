# Running VTKEL System in your local machine

To run **VTKEL system** in your local machine follow below steps carefully.

**Step 1:**

Download the required [files](https://figshare.com/articles/VTKEL_resource_files/8247770/3).

**Step 2:**

Download python source [code](https://github.com/shahidost/Baseline4VTKEL/tree/master/source/code).

**Step 3:**

Save the source code and required files in a directory.

**Step 4:**

Update below python file in your local machine.

1. [main_VT-LinKEr_System.py](https://github.com/shahidost/Baseline4VTKEL/blob/master/source/code/main_VT-LinKEr_System.py)

-	Line 489: insert YAGO taxonomy file path.
-	Line 495: insert insert image directory.
-	Line 508: insert image captions file path.
-	Line 538: insert YOLO pre-train files path.
-	Line 642: to save output images from YOLO object detector insert directory path.
-	Line 645: to save output VT-LinKEr annotations file insert directory path.

2. Run *main_VT-LinKEr_System.py* by using the command for VT-LinKEr:
```
> python main_VT-LinKEr_System.py
```
3. You can see the output .ttl and .jpg files in your local machine respective directory.

4. The debugging/execution of VTKEL system will take some time due to the processing of huge YAGO ontology according to the speed of your local machine.
