Source code for Mini-Project: "Descending commands driving limbed behaviours" for BIOENG-456.

To run the project the architecture the dataset should be downloaded as *CoBar-Dataset* and added to the source folder.
In other words, the required architecture is:
   - CoBar-Dataset/ -> which contains raw data from each strain
   - Meta_Analysis/ -> which contains the code for clustering 
   - Results/ -> which contains "Part_III.py"
   - other files in this repository

Kinematic data analysis can be performed by running "Part_III.py".

Meta_Analysis/: runs the kernel density estimations for the Bonus question in Part 4.

To run a time-point based embedding using T-SNE, run "Python run.py" in the Meta_Analysis/ folder.

To run a fly-based embedding using PCA+GMM, run cells of "fly_based_gmm.ipynb" within the same folder.
