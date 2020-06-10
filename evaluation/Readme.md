This directory consists of python code for the VT-LinKEr evaluation. Download helping [files](https://figshare.com/articles/VTKEL_resource_files/8247770/3) to run the evaluation code.

We evaluated the VTKEL framework for 300, 1000 and 31k+ documents. Details of results can be found in our paper: Coming soon!

### Prerequisites:
- Python 3.6(+) or 2.7(+)
- window or unix machine
- rdflib library 4.2.2(+)
- numpy library 1.16.0(+)
- xml.etree.ElementTree library(+)
- Inernet connection for enabling PIKES tool

### Guidelines:
To run the evaluation code in your local machine please follow the guidelines listed below:

**Step 1:**
Download the required helping files mentioned in YOLO, Gender, OutputImages and VTKELin directory.

**Step 2:**
Save the python files+directory into $root-directory.

**Step 3:**
Update python file in your local machine.

1. [VT-LinKEr_Evaluation.py](https://github.com/shahidost/Baseline4VTKEL/blob/master/evaluation/VT-LinKEr_Evaluation.py)

-	Line 708: update path for YAGO taxonomy .ttl file.
-	Line 715: update image (images should be downloaded from the Flickr30k-entities dataset) directory path.
-	Line 719: updated VTKEL folder path (i.e. VTKEL dataset files).
-	Line 750: insert image captions directory path.
-	Line 781 & 784: update YOLO directory path.
-	Line 903: update OutputImages directory path to save annotated image-files from VT-LinKEr.
-	Line 968: update the directory path to save Turtle annotated file from VT-LinKEr.

2. Run *VT-LinKEr_Evaluation.py* by using the command for Evaluation:
```
> python $root/evaluation/VT-LinKEr_Evaluation.py
```
3. You can find the evaluation results after completing all the files processed.

### Citing:
If you find the Evaluation, VTKEL dataset or VT-LinKEr helpful in your work please cite the papers:
```
@inproceedings{dost2020vtkel,
  title={VTKEL: a resource for visual-textual-knowledge entity linking},
  author={Dost, Shahi and Serafini, Luciano and Rospocher, Marco and Ballan, Lamberto and Sperduti, Alessandro},
  booktitle={Proceedings of the 35th Annual ACM Symposium on Applied Computing},
  pages={2021--2028},
  year={2020}
}

@inproceedings{dost2020visual,
  title={On Visual-Textual-Knowledge Entity Linking},
  author={Dost, Shahi and Serafini, Luciano and Rospocher, Marco and Ballan, Lamberto and Sperduti, Alessandro},
  booktitle={2020 IEEE 14th International Conference on Semantic Computing (ICSC)},
  pages={190--193},
  year={2020},
  organization={IEEE}
}
```
### Note:
If you have any queries regarding Evaluation, dataset, or VT-LinKEr or want to contribute to the system contact on ***sdost[at]fbk.eu***. OR create an issue.
