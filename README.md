# MasterThesis

This repository contains the python code that was produced during my master thesis.

Abstract : 

Radiotherapy targeting mobile tumor is particularly challenging because of the lack of information on the tumor position during treatment. Since 2015 a new medical hardware, MR-Linac, enables the acquisition of live Magnetic Resonance Images (MRI) of the regions targeted during treatment. The provision of live information opens the door to new adaptive radiotherapy allowing, for instance, real-time estimation of the dose deposition. This estimation requires X-Ray-like images, such as Computerised Tomography (CT) scans, which have no direct correlation with the MRI provided by the MR-Linac. Methods have been developed to reproduce the anatomical motion, captured live on MR images, onto images of CT modality. Those methods rely on movement comparison of regions that are similar on each modality. 
This master thesis focuses on the localisation of such regions by using Computer Vision. The method starts by identifying moving edges appearing on both image modalities and for which the movement has a specific direction. To identify such edges, the method works as a funnel. At each step, more and more points are discarded by including new constraints via tools such as Sobel Filters and Dense Displacement Fields. This localisation scheme is used to automate a step of an existing motion replication method that relies on interface tracking. As part of this thesis, a python module was developed that implements the method in that specific context. A validation protocol was developed with a doctor to evaluate the performances of the module. The results are promising.


Any question on this work can be asked by mail at : romain.pattyn.ent@gmail.com
