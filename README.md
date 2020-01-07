# VT-LinKEr: System for Visual-Textual-Knowledge entity linking

### Introduction:
This repository consists of problem, the dataset for the task of **Visual-Textual-Knowledge Entity linking** and the first algorithm called **VT-LinKEr** for solving *VTKEL*. The *VT-LinKEr* is developed by using state-of-the-art tools for object detection using [**YOLO version 3**](https://pjreddie.com/darknet/yolo/), entity recognition and linking to ontologies in text using [**PIKES**](http://pikes.fbk.eu/), and alignment/mapping of visual-textual mentions using [**YAGO knowledgebase**](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago/). The experimental evaluation shows an overall accuracy of **59.7%**. Being VTKEL a new and novel task, the proposed VT-LinKEr can be considered as a first baseline for further improvements. 

### Problem (Visual-Textual-Knowledge Entity Linking):
Given a document *d* composed of a text *d<sub>t</sub>* and an image *d<sub>i</sub>* and a knowledge base **K**, *VTKEL* is the problem of detecting all the entities mentioned in *d<sub>t</sub>* and and/or shown in *d<sub>i</sub>*, and linking them to the corresponding named entities in **K**, if they are present, or linking them to new entities, extending the **A-box** of **K** with its type assertion(s),  i.e. adding *C(e<sup>new</sup>)* for each new entity *C(e<sup>new</sup>)* of type **C** mentioned in *d*.

Consider the document shown in *Figure 1*, which is composed of one picture	and a short sentence (caption) in natural language. As shown in	*Figure 1*, one can find four visual mentions, shown in colored rectangles in the picture, and five textual mentions, underlined in the text. One could find many more visual mentions in the picture (e.g., *racket, ball, logo-on-shirt*) but let us suppose	we are only interested in the mentions of certain types.

<p align="center">
  <img width="450" height="350" src="https://user-images.githubusercontent.com/25593410/59090794-149ea700-890e-11e9-82f7-32e0e0e08222.png">
</p>

The solution of the *VTKEL* task requires:
(i) detecting the visual and	textual entity mentions of the considered types, and linking them to	either (ii) the correct, existing, named entities, or (iii) newly	created entities, adding the corresponding type assertions.

The visual and textual mentions of a *man* shown in the red text and in	the red box refer to the same entity, and they should be linked together. The other visual mention i.e. *racket*, *ball* and *logo* should be linked to different entities. These three	entities are not known (i.e., they are not part of the initial	knowledgebase **K**), and therefore three new entities of type *racket, ball* and *logo* should be added to the knowledge base, i.e., the **A-box** of **K** should be extended with the assertions *Racket(e<sup>new1</sup>)*, *Ball(e<sup>new2</sup>)* and *Logo(e<sup>new3</sup>)*. The visual and textual mentions of *R.Federer* is also referring to the same entity. However, this time the entity is known (i.e., **YAGO** contains an	entity for *man*) and therefore the two mentions should be linked to the same entity.	For the other textual mentions, i.e., *Lukas Lacko*,	*Wimbledon*, *London*, *2018*, we already have instances in the **knowledgebase**, so we	have to link them to these entities. (For details read our papers: coming soon!)

### VT-LinKEr architecture:
*VTKEL* is a multimodal complex problem, which closed the loop between *natural language processing*, *computer vision* and *knowledge representation*. *Figure 2* shows the architecture of the baseline system in detail (the numbering shows the sub-modules of architecture).

<p align="center">
  <img width="800" height="400" src="https://user-images.githubusercontent.com/25593410/59091278-4401e380-890f-11e9-8f5e-80605c0ce831.png">
</p>

### VT-LinKEr annotations instantiantions:
*VT-LinKEr* produced a terse [**RDF**](https://www.w3.org/TR/turtle/) (Resource Description Framework) triple language (Turtle) file (i.e. *.ttl* file) to store the annotations of resultant visual and textual mentions and entity contents (i.e. linking to YAGO Ontology), as well as its links to the region of image and text where it derives from. The annotated file is organized in three distinct yet interlinked representations layers: *Resource, Mention* and *Entity*. The details of instantiantions are showns in *Figure 3* in details for visual mentions *person*, (Complete details can be found in the *Example* folder files).

<p align="center">
  <img width="850" height="600" src="https://user-images.githubusercontent.com/25593410/70332612-4bfcfa80-1842-11ea-8114-a2ccff77dd0c.png">
</p>

### Prerequisites:
- Python 3.6(+) or 2.7(+)
- window or unix machine
- rdflib library 4.2.2(+)
- numpy library 1.16.0(+)
- xml.etree.ElementTree library(+)
- Inernet connection for enabling PIKES tool

### Running VT-LinKEr:
Follow step by step these [guidelines](https://github.com/shahidost/Baseline4VTKEL/tree/master/source) to run VT-LinKEr system in your local machine.

### Quality of VT-LinKEr:
The quality of *VTKEL* system evaluation is shown in below *Table* and for details evaluations please read our paper:

<p align="center">
  <img width="480" height="150" src="https://user-images.githubusercontent.com/25593410/70332969-e3624d80-1842-11ea-840f-a7837ad654f3.png">
</p>

### Citing:
If you find VTKEL problem, datasets, or VT-LinKEr helpful in your work please cite the papers:
```
Shahi Dost, Luciano Serafini, Marco Rospocher, Lamberto Ballan, and Alessandro Sperduti. 2020. 
VTKEL: A resource for Visual-TextualKnowledge Entity Linking. In the 35th ACM/SIGAPP Symposium on
Applied Computing (SAC’20), https://doi.org/10.1145/3341105.3373958 
```


### License:
The VTKEL problem, dataset, VT-LinKEr and their codes are licensed under [CC BY 4.0](https://creativecommons.org/2014/01/07/plaintext-versions-of-creative-commons-4-0-licenses/).

### Contributors:
- [Luciano Serafini](https://dkm.fbk.eu/people/profile/serafini)
- [Alessandro Sperduti](https://www.math.unipd.it/~sperduti/)
- [Marco Rospocher](https://scholar.google.com/citations?user=wkAcWjMAAAAJ&hl=en)
- [Lamberto Ballan](http://www.lambertoballan.net/)
- [Francesco Corcoglioniti](https://scholar.google.com/citations?user=Nw7gPMEAAAAJ&hl=en)

### References:
- [YAGOv3](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago/)
- [YOLOv3](https://pjreddie.com/darknet/yolo/)
- [PIKES](http://pikes.fbk.eu)
- [Flickr30k-Entities](http://bryanplummer.com/Flickr30kEntities/)
- [RDF](https://www.w3.org/TR/turtle/)
- [GenderClassifier](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W08/html/Levi_Age_and_Gender_2015_CVPR_paper.html)


### Contact
If you have any query regarding VTKEL problem, dataset, or VT-LinKEr or want to contribute to the system contact on ***sdost[at]fbk.eu***.
