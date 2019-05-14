# bbox_from_segmentation
generate rotated bounding box single object from binary segmentation mask (supplementary code for VOT Challenge data generation)

# DATA preparation

binary segmentation images in dir strucure e.g.:
DATA  
├── segmentations  
│   ├── sequence_1  
│   │   └── 0000.png  
│   │   └── 0001.png  
│   │		.  
│   │		.  
│   │		.  	 
  
rgb images in dir strucure e.g.:  
DATA  
├── sequences  
│   ├── sequence_1  
│   │   └── groundtruth.txt (ground truth bounding boxes as x,y,w,h)  
│   │   └── 0000.png  
│   │   └── 0001.png  
│   │		.  
│   │		.  
│   │		.	  

fill these dir path to `segmentation_dirs` and `sequences_dirs` vars in process_seqs.m
and create file list.txt containing "sequence_1"

# RUN

$matlab -nodisplay -nosplash -r "process_seqs('list.txt');" &>log-list.txt