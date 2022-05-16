# C. elegans connectomes
This data came from worm wiring https://www.wormwiring.org/pages/adjacency.html
The file accessed was "SI 5 Connectome adjacency matrices, corrected July 2020.xlsx"
which originated from Cook et al. *Nature* 2019. `herm_chem_adj.csv` and 
`male_chem_adj.csv` were created as copies of the respective sheets in the Excel
workbook above to make them easier to parse; all other modifications were done
programatically (see `scripts/process_c_elegans.py`).
  