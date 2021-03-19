Routines


main-global-v.2.5.py: main program implementing one global neural classifier and leave-one-out patient (LOOP) evaluation

config.py: Contains our configuration of important variables, including image and batch size, epochs, input and output file paths, etc.

ceusutils: Includes helper utilities, including a function to implement new LOOP Generator, custom metrics, a training history and statistics plotting function

model: Includes custom and classic NNs


Version History

v1.0 - all images, various splits

V2.0 - one-out patient, log the experiments, statistics

v2.1 - shuffle patterns, multiple experiments

v2.2 - memory leaks solved (Resource exhausted:  Failed to allocate memory for the batch)

v2.3 - max voting for validation error

v2.4 - introduce pre-trained/transfer learning DNN support, enhanced results display & statistics   

v.2.5 - globals to config.py, interface to TensorBoard, added color_mode "rbg"/"rgba"/"grayscale", enhance log/output, 4 custom,  4 pre-trained 


Fetaures TBD

- display % of total done
- estimate remaining time
- log training history, settings (model name, patinets taken, etc)
- leave out specific patient
- opt between leave one out and split evaluations
